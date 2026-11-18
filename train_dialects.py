"""Multi-dialect / new-language fine-tuning for Qwen3-TTS (12Hz, Base checkpoints).

Based on the official finetuning/sft_12hz.py + dataset.py, with:
  * label-shift fix (issue #179 / PR #178): CE computed directly on logits, single shift
  * text_projection applied to text embeddings when present (PR #188)
  * fp32 master weights + bf16 autocast (1e-6-scale updates don't vanish)
  * AdamW + warmup + cosine schedule (stepped once per optimizer step), grad clip 1.0,
    sub-talker loss x0.3
  * one NEW codec language ID per dialect in the "think" prefix:
        [think, think_bos, <LANG>, think_eos, <SPK>, codec_pad]   (text offset 8 -> 9)
    learned as a small separate embedding with its own (higher) LR, warm-started from the
    mean of the existing language embeddings + noise, written into codec_embedding on save
  * language-ID dropout -> the sample uses the stock "auto" prefix
  * per-row ref_audio (speaker encoder frozen), length-bucketed batches
  * multi-GPU (accelerate launch), resumable (--resume 1), codes held as int16 (low RAM)
  * validation loss, throughput + ETA logging, mid-epoch checkpoints
  * saves a BASE-type checkpoint (keeps speaker encoder: voice cloning + continued training)
    and optionally a CUSTOM_VOICE export with several speakers (rows 3000, 3001, ...)

Run `python train_dialects.py --train_jsonl x --check_only` first: it prints the token IDs it
will use and checks the new rows are free in your qwen-tts version.
"""
import argparse, json, math, os, random, shutil, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.environ.get("QWEN3_TTS_DIR", os.path.join(HERE, "Qwen3-TTS"))
sys.path.insert(0, os.path.join(QDIR, "finetuning"))

DEFAULT_DIALECTS = "ar_msa,ar_egy,ar_sau,ar_uae,ar_gulf,ar_omn,ar_lev,ar_irq,ar_mar,ar_tun,ar_dza"
FRAME_HZ = 12.5


# ----------------------------------------------------------------------------- utils
def tc_get(tc, name):
    v = getattr(tc, name, None)
    if v is None and isinstance(tc, dict):
        v = tc.get(name)
    if v is None:
        raise KeyError(f"talker_config has no '{name}' - check your qwen-tts version")
    return v


def reserved_codec_ids(tc):
    ids = set()
    d = tc.to_dict() if hasattr(tc, "to_dict") else dict(tc)
    for k, v in d.items():
        if isinstance(v, int) and (k.endswith("_id") or k.endswith("_token_id")):
            ids.add(v)
        if k in ("codec_language_id", "spk_id") and isinstance(v, dict):
            ids.update(int(x) for x in v.values())
    return ids


def data_root():
    return os.environ.get("QWEN_DATA_ROOT", "/data").rstrip("/") or "/data"


def remap_data(path):
    """Volume jsonl stores absolute /data paths. Local runs point those at QWEN_DATA_ROOT.

    The default root is /data, so a Modal job reads the same paths it always did.
    """
    if not isinstance(path, str):
        return path
    root = data_root()
    if path == "/data" or path.startswith("/data/"):
        return root + path[len("/data"):]
    return path


def resolve_local(path):
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path)


# ----------------------------------------------------------------------------- data
def build_dataset_cls():
    from dataset import TTSDataset  # official

    class DialectTTSDataset(TTSDataset):
        def __init__(self, rows, processor, config, dialect2idx, id_dropout, max_frames, diac_prob=0.0):
            super().__init__(rows, processor, config)
            self.dialect2idx = dialect2idx
            self.id_dropout = id_dropout
            self.max_frames = max_frames
            self.diac_prob = diac_prob
            tc = config.talker_config
            self.T = dict(
                think=tc_get(tc, "codec_think_id"), think_bos=tc_get(tc, "codec_think_bos_id"),
                think_eos=tc_get(tc, "codec_think_eos_id"), nothink=tc_get(tc, "codec_nothink_id"),
                pad=tc_get(tc, "codec_pad_id"), bos=tc_get(tc, "codec_bos_id"),
                eos=tc_get(tc, "codec_eos_token_id"),
            )

        def __getitem__(self, idx):
            it = self.data_list[idx]
            saved = it.get("text")
            used = 0
            if self.diac_prob > 0 and not it.get("no_diac") and random.random() < self.diac_prob:
                vow = it.get("text_diac2") or it.get("text_diac") or ""
                if vow:
                    it["text"] = vow
                    used = 1
            try:
                for attempt in range(5):
                    try:
                        d = super().__getitem__(idx)
                        break
                    except OSError:
                        if attempt == 4:
                            raise
                        time.sleep(2 * (attempt + 1))
            finally:
                it["text"] = saved
            it = self.data_list[idx]
            li = self.dialect2idx.get(it.get("dialect"), -1)
            if li >= 0 and self.id_dropout > 0 and random.random() < self.id_dropout:
                li = -1
            d["lang_idx"] = li
            d["speaker"] = str(it.get("speaker", it.get("dialect", "spk")))
            d["used_diac"] = used
            if d["audio_codes"].shape[0] > self.max_frames:
                d["audio_codes"] = d["audio_codes"][: self.max_frames]
            return d

        def collate_fn(self, batch):
            cfg, T = self.config, self.T
            b = len(batch)
            t = max(x["text_ids"].shape[1] + x["audio_codes"].shape[0] for x in batch) + 9
            input_ids = torch.zeros((b, t, 2), dtype=torch.long)
            codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
            text_mask = torch.zeros((b, t), dtype=torch.bool)
            cemb_mask = torch.zeros((b, t), dtype=torch.bool)
            codec_mask = torch.zeros((b, t), dtype=torch.bool)
            attn = torch.zeros((b, t), dtype=torch.long)
            labels = torch.full((b, t), -100, dtype=torch.long)
            spk_pos = torch.zeros(b, dtype=torch.long)
            lang_pos = torch.full((b,), -1, dtype=torch.long)
            lang_idx = torch.full((b,), -1, dtype=torch.long)

            for i, x in enumerate(batch):
                txt, codes = x["text_ids"], x["audio_codes"]
                L, C = txt.shape[1], codes.shape[0]
                use_lang = x["lang_idx"] >= 0
                o = 9 if use_lang else 8          # first text-token position
                # text channel
                input_ids[i, :3, 0] = txt[0, :3]
                input_ids[i, 3:o - 1, 0] = cfg.tts_pad_token_id
                input_ids[i, o - 1, 0] = cfg.tts_bos_token_id
                input_ids[i, o:o + L - 3, 0] = txt[0, 3:]
                input_ids[i, o + L - 3, 0] = cfg.tts_eos_token_id
                input_ids[i, o + L - 2:o + L + C, 0] = cfg.tts_pad_token_id
                text_mask[i, :o + L + C] = True
                # codec channel prefix
                if use_lang:
                    prefix = [T["think"], T["think_bos"], 0, T["think_eos"], 0, T["pad"]]
                    lang_pos[i], lang_idx[i], spk_pos[i] = 5, x["lang_idx"], 7
                else:
                    prefix = [T["nothink"], T["think_bos"], T["think_eos"], 0, T["pad"]]
                    spk_pos[i] = 6
                input_ids[i, 3:o, 1] = torch.tensor(prefix)
                input_ids[i, o:o + L - 2, 1] = T["pad"]
                input_ids[i, o + L - 2, 1] = T["bos"]
                input_ids[i, o + L - 1:o + L - 1 + C, 1] = codes[:, 0]
                input_ids[i, o + L - 1 + C, 1] = T["eos"]
                labels[i, o + L - 1:o + L - 1 + C] = codes[:, 0]
                labels[i, o + L - 1 + C] = T["eos"]
                codec_ids[i, o + L - 1:o + L - 1 + C, :] = codes
                cemb_mask[i, 3:o + L + C] = True
                cemb_mask[i, spk_pos[i]] = False
                if use_lang:
                    cemb_mask[i, 5] = False
                codec_mask[i, o + L - 1:o + L - 1 + C] = True
                attn[i, :o + L + C] = 1

            ref_mels = []
            for x in batch:
                m = x["ref_mel"]
                if m.dim() == 3:
                    m = m.squeeze(0)
                ref_mels.append(m)

            return dict(input_ids=input_ids, codec_ids=codec_ids,
                        ref_mels=ref_mels,
                        text_embedding_mask=text_mask.unsqueeze(-1),
                        codec_embedding_mask=cemb_mask.unsqueeze(-1),
                        codec_mask=codec_mask, attention_mask=attn, codec_0_labels=labels,
                        spk_pos=spk_pos, lang_pos=lang_pos, lang_idx=lang_idx,
                        speakers=[x["speaker"] for x in batch],
                        used_diac=torch.tensor([x.get("used_diac", 0) for x in batch], dtype=torch.float))

    return DialectTTSDataset


def bucket_batches(lengths, batch_size, seed, mega=100):
    rng = random.Random(seed)
    idx = list(range(len(lengths))); rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), batch_size * mega):
        chunk = sorted(idx[s:s + batch_size * mega], key=lambda i: lengths[i])
        batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def load_rows(path, require_split=""):
    """Codes are converted to int16 arrays while reading: ~2 bytes/code instead of ~36.

    require_split drops rows that belong to another split. A row with no split field
    is kept, so the older validation files still load.
    """
    rows = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for key, val in list(r.items()):
                if isinstance(val, str):
                    r[key] = remap_data(val)
                elif isinstance(val, list):
                    r[key] = [remap_data(item) if isinstance(item, str) else item for item in val]
            if require_split and r.get("split") and r.get("split") != require_split:
                continue
            if require_split and (not r.get("audio_codes") or not str(r.get("text") or "").strip()):
                skipped += 1
                continue
            if "audio_codes" in r:
                r["audio_codes"] = np.asarray(r["audio_codes"], dtype=np.int16)
            rows.append(r)
    if skipped:
        print(f"skipped {skipped} rows without codes or text", flush=True)
    return rows


def apply_cond(rows, cond):
    """Text-tag conditioning: tag prepended to the text; use the SAME tag at inference."""
    if cond in ("text_tag", "both"):
        for r in rows:
            if r.get("dialect") and not r["text"].startswith("<"):
                r["text"] = f"<{r['dialect']}> {r['text']}"
    return rows


def make_lang_emb(codec_weight, talker_config, dialects, lang_id_start, seed, init_noise, vocab):
    """Same language-row init the trainer uses. Existing ids are copied; new ids start from the mean."""
    existing = dict(getattr(talker_config, "codec_language_id", None) or {})
    reserved = reserved_codec_ids(talker_config)
    base_lang_ids = [v for k, v in existing.items() if k not in dialects]
    lang_map, nxt = {}, lang_id_start
    for d in dialects:
        if d in existing:
            lang_map[d] = int(existing[d])
        else:
            while nxt in reserved or nxt in lang_map.values():
                nxt += 1
            lang_map[d] = nxt
            nxt += 1
    for d, cid in lang_map.items():
        assert cid < vocab, f"id {cid} >= codec vocab {vocab}"
        assert d in existing or cid not in reserved, f"id {cid} for {d} collides with a reserved id"
    W = codec_weight.detach().float().cpu()
    D = W.shape[1]
    mu = W[base_lang_ids].mean(0) if base_lang_ids else W[:vocab].mean(0)
    g = torch.Generator().manual_seed(seed)
    init = []
    for d in dialects:
        if d in existing:
            init.append(W[lang_map[d]])
        else:
            init.append(mu + torch.randn(D, generator=g) * init_noise * mu.norm() / math.sqrt(D))
    lang_emb = nn.Embedding(len(dialects), D)
    lang_emb.weight.data.copy_(torch.stack(init))
    return lang_emb, lang_map, existing
def _ce(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def speaker_embeds(model, mels, dev):
    """One embedding per clip, on that clip's mel only. Padding would change the vector."""
    dtype = next(model.speaker_encoder.parameters()).dtype
    out = []
    for m in mels:
        if m.dim() == 3:
            m = m.squeeze(0)
        e = model.speaker_encoder(m.unsqueeze(0).to(device=dev, dtype=dtype))
        out.append(e.reshape(-1))
    return torch.stack(out, 0)


def talker_step(model, lang_emb, batch):
    """One teacher-forced talker pass. Returns the tensors every alignment is scored on."""
    dev = next(model.talker.parameters()).device
    ids = batch["input_ids"].to(dev)
    b = ids.shape[0]
    bi = torch.arange(b, device=dev)
    with torch.no_grad():
        spk = speaker_embeds(model, batch["ref_mels"], dev)
    txt = model.talker.model.text_embedding(ids[:, :, 0])
    if hasattr(model.talker, "text_projection"):
        txt = model.talker.text_projection(txt)
    txt = txt * batch["text_embedding_mask"].to(dev)
    cod = model.talker.model.codec_embedding(ids[:, :, 1]) * batch["codec_embedding_mask"].to(dev)
    cod = cod.clone()
    cod[bi, batch["spk_pos"].to(dev)] = spk.to(cod.dtype)
    lidx = batch["lang_idx"].to(dev)
    has = lidx >= 0
    if has.any():
        cod[bi[has], batch["lang_pos"].to(dev)[has]] = lang_emb(lidx[has]).to(cod.dtype)
    emb = txt + cod
    codec_ids, cmask = batch["codec_ids"].to(dev), batch["codec_mask"].to(dev)
    for i in range(1, 16):
        e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        emb = emb + e * cmask.unsqueeze(-1)

    out = model.talker(inputs_embeds=emb[:, :-1], attention_mask=batch["attention_mask"].to(dev)[:, :-1],
                       output_hidden_states=True)
    logits = out.logits.float()
    labels = batch["codec_0_labels"].to(dev)
    hs = out.hidden_states[0][-1]
    return logits, labels, hs, codec_ids, cmask, spk


def sub_scores(model, codec_ids, cmask, hidden):
    """Score codebooks 1..15 from one hidden-state choice.

    The package loss shifts labels that are already aligned, so the recipe uses
    a plain cross-entropy on the returned logits instead.
    """
    ids = codec_ids[cmask]
    if hidden.shape[0] != ids.shape[0]:
        raise RuntimeError(f"sub-talker frames {tuple(ids.shape)} vs hidden {tuple(hidden.shape)}")
    sub_logits, package = model.talker.forward_sub_talker_finetune(ids, hidden)
    manual = _ce(sub_logits.float(), ids[:, 1:])
    return manual, package


def alignment_table(model, lang_emb, batch):
    logits, labels, hs, codec_ids, cmask, _spk = talker_step(model, lang_emb, batch)
    main = {
        "single shift (recipe)": _ce(logits, labels[:, 1:]),
        "no shift": _ce(logits, labels[:, :-1]),
        "double shift": _ce(logits[:, :-1], labels[:, 2:]),
    }
    prev_manual, prev_package = sub_scores(model, codec_ids, cmask, hs[cmask[:, 1:]])
    frame_manual, frame_package = sub_scores(model, codec_ids, cmask, hs[cmask[:, :-1]])
    sub = {
        "prev / no shift (recipe)": prev_manual,
        "prev / package shift": prev_package,
        "at-frame / no shift": frame_manual,
        "at-frame / package shift": frame_package,
    }
    return main, sub


def forward_loss(model, lang_emb, batch, sub_w):
    logits, labels, hs, codec_ids, cmask, spk = talker_step(model, lang_emb, batch)
    main = _ce(logits, labels[:, 1:])  # position t predicts token t+1 (single shift)
    # Previous step's hidden state: the one that predicted this frame's codebook 0.
    sub, _package = sub_scores(model, codec_ids, cmask, hs[cmask[:, 1:]])
    return main + sub_w * sub, main.detach(), sub.detach(), spk.detach()


class TTSWrapper(nn.Module):
    """One module so DDP sees the whole forward (needed for correct multi-GPU gradient sync)."""

    def __init__(self, model, lang_emb, sub_w):
        super().__init__()
        self.model, self.lang_emb, self.sub_w = model, lang_emb, sub_w

    def forward(self, batch):
        return forward_loss(self.model, self.lang_emb, batch, self.sub_w)


# ----------------------------------------------------------------------------- save
def save_checkpoint(wrapper, init_dir, out_dir, lang_map, spk_stats, args, tag, log=print):
    m, lang_emb = wrapper.model, wrapper.lang_emb
    W = lang_emb.weight.detach().float().cpu()
    dtype = torch.bfloat16
    sd = {k: (v.detach().to("cpu").to(dtype) if v.is_floating_point() else v.detach().to("cpu"))
          for k, v in m.state_dict().items()}
    key = "talker.model.codec_embedding.weight"
    for j, (name, cid) in enumerate(lang_map.items()):
        sd[key][cid] = W[j].to(dtype)
    cfg = json.load(open(os.path.join(init_dir, "config.json"), encoding="utf-8"))
    tcfg = cfg.setdefault("talker_config", {})
    tcfg.setdefault("codec_language_id", {}).update(lang_map)

    # 1) BASE-type checkpoint (keeps speaker encoder; use for Stage B init + voice cloning)
    bdir = os.path.join(out_dir, f"{tag}-base")
    shutil.copytree(init_dir, bdir, dirs_exist_ok=True)
    json.dump(cfg, open(os.path.join(bdir, "config.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    save_file(sd, os.path.join(bdir, "model.safetensors"))

    # 2) CUSTOM_VOICE export with top-N speakers (mean embedding over their training clips)
    if args.export_speakers > 0 and spk_stats:
        top = sorted(spk_stats.items(), key=lambda kv: -kv[1][1])[: args.export_speakers]
        vdir = os.path.join(out_dir, f"{tag}-voices")
        shutil.copytree(init_dir, vdir, dirs_exist_ok=True)
        c2 = json.loads(json.dumps(cfg))
        c2["tts_model_type"] = "custom_voice"
        t2 = c2["talker_config"]
        t2["spk_id"], t2["spk_is_dialect"] = {}, {}
        sd2 = {k: v for k, v in sd.items() if not k.startswith("speaker_encoder")}
        sd2[key] = sd2[key].clone()
        for j, (spk, (s, n)) in enumerate(top):
            row = args.speaker_row_start + j
            name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in spk)
            t2["spk_id"][name] = row
            t2["spk_is_dialect"][name] = False
            sd2[key][row] = (s / n).to(dtype)
        json.dump(c2, open(os.path.join(vdir, "config.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        save_file(sd2, os.path.join(vdir, "model.safetensors"))
    log(f"saved {tag}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--train_jsonl", required=True, help="rows with audio_codes (prepare_codes.sh)")
    ap.add_argument("--val_jsonl", default=None)
    ap.add_argument("--extra_val_jsonl", default="", help="second val set; printed, not used by the stop rule")
    ap.add_argument("--extra_val_jsonl2", default="", help="third val set; printed, not used by the stop rule")
    ap.add_argument("--require_split", default="", help="train on this split only; rows with no split field stay")
    ap.add_argument("--diac_prob", type=float, default=0.0, help="chance to train on text_diac2, else text_diac, else plain")
    ap.add_argument("--output_dir", default="runs/exp")
    ap.add_argument("--dialects", default=DEFAULT_DIALECTS)
    ap.add_argument("--lang_id_start", type=int, default=2072)
    ap.add_argument("--speaker_row_start", type=int, default=3000)
    ap.add_argument("--export_speakers", type=int, default=0)
    ap.add_argument("--id_dropout", type=float, default=0.1)
    ap.add_argument("--init_noise", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lang_lr_mult", type=float, default=20.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch_size", type=int, default=8, help="per GPU")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--sub_talker_weight", type=float, default=0.3)
    ap.add_argument("--max_frames", type=int, default=375, help="30 s at 12.5 Hz")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--val_batches", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=0, help="optimizer steps; 0 = half-epoch")
    ap.add_argument("--save_at", default="", help="extra steps to save, comma-separated")
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2")
    ap.add_argument("--fp32_master", type=int, default=1)
    ap.add_argument("--freeze_code_predictor", type=int, default=0)
    ap.add_argument("--cond", default="id", choices=["id", "text_tag", "both"],
                    help="codec language IDs | dialect tag in text ('<ar_egy> ...') | both")
    ap.add_argument("--resume", type=int, default=1, help="resume from <output_dir>/state if present")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--eval_repeats", type=int, default=0,
                    help="plain-text val passes before any step; 0 skips them")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="stop after this many optimizer steps; 0 runs the full epoch count")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    acc = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16",
                      log_with="tensorboard", project_dir=args.output_dir,
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    init_dir = resolve_local(args.init_model_path)
    tts = Qwen3TTSModel.from_pretrained(init_dir, torch_dtype=torch.float32 if args.fp32_master else torch.bfloat16,
                                        attn_implementation=args.attn)
    model = tts.model
    config = AutoConfig.from_pretrained(init_dir)
    tc = config.talker_config
    emb = model.talker.model.codec_embedding
    vocab, D = emb.num_embeddings, emb.embedding_dim

    # ---- language IDs: reuse if init already has them (Stage B), else allocate new rows
    dialects = [d.strip() for d in args.dialects.split(",") if d.strip()]
    lang_emb, lang_map, existing = make_lang_emb(
        emb.weight, tc, dialects, args.lang_id_start, args.seed, args.init_noise, vocab)
    reserved = reserved_codec_ids(tc)
    spk_rows = range(args.speaker_row_start, args.speaker_row_start + args.export_speakers)
    bad = [r for r in spk_rows if r >= vocab or (r in reserved and r not in (getattr(tc, "spk_id", {}) or {}).values())]
    if bad:
        acc.print(f"WARNING speaker rows {bad[:5]}... invalid/reserved; lower --export_speakers")

    acc.print(f"codec vocab={vocab} dim={D}")
    acc.print(f"base language ids: {existing}")
    acc.print(f"dialect ids: {lang_map}")
    acc.print(f"special: think={tc_get(tc,'codec_think_id')} think_bos={tc_get(tc,'codec_think_bos_id')} "
              f"think_eos={tc_get(tc,'codec_think_eos_id')} nothink={tc_get(tc,'codec_nothink_id')} "
              f"pad={tc_get(tc,'codec_pad_id')} bos={tc_get(tc,'codec_bos_id')} eos={tc_get(tc,'codec_eos_token_id')}")
    acc.print(f"has text_projection: {hasattr(model.talker, 'text_projection')}")
    if args.check_only:
        return

    # ---- data
    DS = build_dataset_cls()
    d2i = {d: i for i, d in enumerate(dialects)}
    if args.cond == "text_tag":
        d2i = {}  # no codec language IDs: stock "auto" prefix + tag in the text
    t_load = time.time()
    tr_rows = apply_cond(load_rows(args.train_jsonl, args.require_split), args.cond)
    ds = DS(tr_rows, tts.processor, config, d2i, args.id_dropout, args.max_frames, args.diac_prob)
    lengths = [len(r["audio_codes"]) + len(r["text"]) // 2 for r in tr_rows]
    miss = sum(r.get("dialect") not in d2i for r in tr_rows)
    hours = sum(len(r["audio_codes"]) for r in tr_rows) / FRAME_HZ / 3600
    acc.print(f"train rows {len(tr_rows):,d} ({hours:.1f} h) loaded in {time.time()-t_load:.0f}s | "
              f"rows trained as auto (unknown dialect): {miss}")
    val_dl = None
    if args.val_jsonl:
        vrows = apply_cond(load_rows(args.val_jsonl), args.cond)
        vds = DS(vrows, tts.processor, config, d2i, 0.0, args.max_frames, 0.0)
        val_dl = DataLoader(vds, batch_size=args.batch_size, shuffle=False, collate_fn=vds.collate_fn, num_workers=2)
    extra_dl = None
    if args.extra_val_jsonl:
        erows = apply_cond(load_rows(args.extra_val_jsonl), args.cond)
        eds = DS(erows, tts.processor, config, d2i, 0.0, args.max_frames, 0.0)
        extra_dl = DataLoader(eds, batch_size=args.batch_size, shuffle=False, collate_fn=eds.collate_fn, num_workers=2)
    extra2_dl = None
    if args.extra_val_jsonl2:
        e2rows = apply_cond(load_rows(args.extra_val_jsonl2), args.cond)
        e2ds = DS(e2rows, tts.processor, config, d2i, 0.0, args.max_frames, 0.0)
        extra2_dl = DataLoader(e2ds, batch_size=args.batch_size, shuffle=False, collate_fn=e2ds.collate_fn, num_workers=2)

    # ---- params / optimizer / schedule
    for p in model.speaker_encoder.parameters():
        p.requires_grad = False
    if args.freeze_code_predictor:
        for p in model.talker.code_predictor.parameters():
            p.requires_grad = False
    wrapper = TTSWrapper(model, lang_emb, args.sub_talker_weight)
    main_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": list(lang_emb.parameters()), "lr": args.lr * args.lang_lr_mult, "weight_decay": 0.0},
    ], betas=(0.9, 0.95), eps=1e-8)

    world = acc.num_processes
    n_batches = math.ceil(len(tr_rows) / args.batch_size)
    per_rank = (n_batches // world) // args.grad_accum * args.grad_accum
    steps_per_epoch = per_rank // args.grad_accum
    assert steps_per_epoch > 0, "not enough data for one optimizer step per rank"
    total = int(steps_per_epoch * args.epochs)
    # scheduler is NOT passed to accelerate.prepare (that would step it num_processes times per step)
    sched = get_cosine_schedule_with_warmup(opt, int(total * args.warmup_frac), total)
    save_every = args.save_every or max(1, steps_per_epoch // 2)
    save_at = set()
    for part in str(args.save_at or "").split(","):
        if part.strip():
            save_at.add(int(part.strip()))
    wrapper, opt = acc.prepare(wrapper, opt)
    acc.register_for_checkpointing(sched)
    if val_dl is not None:
        val_dl = acc.prepare(val_dl)
    if extra_dl is not None:
        extra_dl = acc.prepare(extra_dl)
    if extra2_dl is not None:
        extra2_dl = acc.prepare(extra2_dl)
    acc.init_trackers("tts")
    acc.print(f"GPUs {world} | per-GPU batch {args.batch_size} x accum {args.grad_accum} "
              f"= {args.batch_size*args.grad_accum*world} clips/step | steps/epoch {steps_per_epoch} | "
              f"total {total} | save every {save_every} | save at {sorted(save_at)}")

    # ---- resume
    # state is written to state.tmp, then swapped in, so a crash mid-save never corrupts it
    state_dir = os.path.join(args.output_dir, "state")
    step, ep, skip = 0, 0, 0
    if args.resume:
        for cand in (state_dir, state_dir + ".old"):
            ts_path = os.path.join(cand, "trainer_state.json")
            if os.path.exists(ts_path):
                acc.load_state(cand)
                ts = json.load(open(ts_path))
                step, ep, skip = ts["step"], ts["epoch"], ts["step_in_epoch"]
                acc.print(f"RESUMED from {cand}: step {step}, epoch {ep}, step_in_epoch {skip}")
                break

    spk_stats = {}
    last_grad = None
    core = acc.unwrap_model(wrapper)
    t0, steps0, audio_s = time.time(), step, 0.0
    diac_seen = diac_tot = 0.0

    def eval_main(loader):
        vals = []
        with torch.no_grad():
            for k, vb in enumerate(loader):
                if k >= args.val_batches:
                    break
                with acc.autocast():
                    _, vm, _, _ = core(vb)
                vals.append(vm.reshape(1))
        if not vals:
            return None
        return acc.gather(torch.cat(vals).mean().reshape(1)).mean().item()

    def weight_abs_sum():
        total = None
        with torch.no_grad():
            for p in core.parameters():
                part = p.detach().float().abs().sum()
                total = part if total is None else total + part
        return float(total)

    if args.eval_repeats and val_dl is not None:
        wrapper.eval()
        for i in range(1, args.eval_repeats + 1):
            vl = eval_main(val_dl)
            acc.print(f"CHECK val {i} {vl:.4f}")
        acc.print(f"CHECK weights {weight_abs_sum():.4f}")
        if args.max_steps <= 0:
            return
        wrapper.train()

    def save_all(tag, step_in_epoch):
        tmp, old = state_dir + ".tmp", state_dir + ".old"
        acc.wait_for_everyone()
        if acc.is_main_process and os.path.exists(tmp):
            shutil.rmtree(tmp)
        acc.wait_for_everyone()
        acc.save_state(tmp)
        acc.wait_for_everyone()
        if acc.is_main_process:
            json.dump({"step": step, "epoch": ep, "step_in_epoch": step_in_epoch},
                      open(os.path.join(tmp, "trainer_state.json"), "w"))
            if os.path.exists(old):
                shutil.rmtree(old)
            if os.path.exists(state_dir):
                os.rename(state_dir, old)
            os.rename(tmp, state_dir)
            shutil.rmtree(old, ignore_errors=True)
            save_checkpoint(core, init_dir, args.output_dir, lang_map, spk_stats, args, tag, acc.print)
        acc.wait_for_everyone()

    wrapper.train()
    halt = False
    while step < total and not halt:
        batches = bucket_batches(lengths, args.batch_size, args.seed + ep)
        mine = batches[acc.process_index::world][:per_rank]
        mine = mine[skip * args.grad_accum:]
        step_in_epoch, skip = (skip, 0)
        dl = DataLoader(ds, batch_sampler=mine, collate_fn=ds.collate_fn,
                        num_workers=args.num_workers, pin_memory=True)
        for batch in dl:
            with acc.accumulate(wrapper):
                loss, lm, ls, spk = wrapper(batch)
                acc.backward(loss)
                if acc.sync_gradients:
                    last_grad = float(acc.clip_grad_norm_(wrapper.parameters(), 1.0))
                opt.step(); opt.zero_grad()
            audio_s += batch["codec_mask"].sum().item() / FRAME_HZ
            if acc.is_main_process and args.export_speakers > 0:
                for s_name, e in zip(batch["speakers"], spk.float().cpu()):
                    s, n = spk_stats.get(s_name, (torch.zeros_like(e), 0))
                    spk_stats[s_name] = (s + e, n + 1)
            if not acc.sync_gradients:
                continue
            sched.step()
            step += 1; step_in_epoch += 1
            if step == 1 and acc.is_main_process:
                acc.print(f"ep {ep} step {step}/{total} started")
            if args.diac_prob > 0 and acc.is_main_process:
                diac_seen += float(batch["used_diac"].sum())
                diac_tot += float(batch["used_diac"].numel())
            if step % 20 == 0:
                el = time.time() - t0
                rate = (step - steps0) / max(el, 1e-6)
                eta_h = (total - step) / max(rate, 1e-9) / 3600
                ah_per_h = audio_s * world / 3600 / max(el / 3600, 1e-9)
                log_row = {"loss": loss.item(), "main": lm.item(), "sub": ls.item(),
                           "lr": sched.get_last_lr()[0]}
                gnote = ""
                if last_grad is not None:
                    log_row["grad_norm"] = last_grad
                    gnote = f" grad {last_grad:.2f}"
                acc.log(log_row, step=step)
                diac_note = ""
                if args.diac_prob > 0:
                    diac_note = f" diac {diac_seen / max(diac_tot, 1):.2f}"
                acc.print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} main {lm.item():.4f} "
                          f"sub {ls.item():.4f} lr {sched.get_last_lr()[0]:.2e} | "
                          f"{ah_per_h:.0f} audio-h/h | ETA {eta_h:.1f} h{diac_note}{gnote}")
            if val_dl is not None and step % args.eval_every == 0:
                wrapper.eval()
                vl = eval_main(val_dl)
                if vl is not None:
                    acc.log({"val_main": vl}, step=step); acc.print(f"  val main loss {vl:.4f}")
                if extra_dl is not None:
                    xl = eval_main(extra_dl)
                    if xl is not None:
                        acc.log({"val_extra_main": xl}, step=step)
                        acc.print(f"  val extra main loss {xl:.4f}")
                if extra2_dl is not None:
                    x2 = eval_main(extra2_dl)
                    if x2 is not None:
                        acc.log({"val_extra2_main": x2}, step=step)
                        acc.print(f"  val extra2 main loss {x2:.4f}")
                wrapper.train()
            if step in save_at or step % save_every == 0 or step == total:
                save_all(f"step-{step:06d}", step_in_epoch)
            if args.max_steps and step >= args.max_steps:
                acc.print(f"CHECK weights after {weight_abs_sum():.4f}")
                halt = True
                break
            if step >= total:
                break
        ep += 1
    acc.end_training()
    acc.print("training finished")


if __name__ == "__main__":
    main()
