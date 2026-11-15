"""Command line interface: ``tujia-asr AUDIO [options]``."""

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .pipeline import recognize


def _format(label, fmt):
    if fmt == "ascii":
        return label
    try:
        from tujia_nlp import format_syllable

        return format_syllable(label, fmt)
    except Exception:
        return label


def _print_report(res, fmt):
    audio = res["audio"]
    lat = res["frontends"]["lattice"]
    print(
        f"# {Path(audio['path']).name}  {audio['dur']}s  "
        f"{lat['backend']}  {lat['n_spikes']} spikes  "
        f"tier-A {res['coverage']['tier_a_ratio']:.0%}"
    )
    if res["words"]:
        print("\n=== words")
        for w in res["words"]:
            han = f" 「{w['han']}」" if w.get("han") else ""
            print(f"[{w['t0']:7.2f}→{w['t1']:7.2f}] {w['ortho']}{han}  {w.get('tier', '')}")
    else:
        print("\n=== syllables")
        for s in res["syllables"]:
            print(
                f"[{s['t0']:7.2f}→{s['t1']:7.2f}] {_format(s['ipa'], fmt):<10} "
                f"tone={s.get('tone')}  conf={s['tone_conf']:.2f}"
            )
    # `text` is a space separated syllable string, so it has to go through the
    # same formatter as the per-syllable lines. Otherwise `--format ye` showed
    # Ye syllables but an IPA summary right below them.
    text = " ".join(_format(tok, fmt) for tok in str(res["text"]).split())
    print(f"\ntext: {text}")
    if res["warnings"]:
        print("\n=== warnings")
        for w in res["warnings"]:
            print(f"  - {w}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="tujia-asr",
        description="Tujia speech recognition (syllable CTC + optional lexicon word decoding)",
    )
    ap.add_argument("wav", help="16 kHz mono audio; other sample rates are resampled")
    ap.add_argument("--dict", dest="dicts", action="append", default=[],
                    help="lexicon JSONL (repeatable); rows are added to the built-in pack")
    ap.add_argument("--no-default-dict", dest="no_default_dict", action="store_true",
                    help="do not load the tujia-nlp open core vocabulary")
    ap.add_argument("--out", help="write the full JSON result to this path")
    ap.add_argument("--json", nargs="?", const="-", default=None,
                    help="write JSON (default: stdout)")
    ap.add_argument("--format", choices=["ipa", "ascii", "ye"], default="ipa",
                    help="display format for syllables")
    ap.add_argument("--tone", dest="tone", action="store_true", default=True)
    ap.add_argument("--no-tone", dest="tone", action="store_false",
                    help="skip F0 tone evidence (the CTC tone head still runs)")
    ap.add_argument("--frontend", choices=["ctc", "zipa"], default=None)
    ap.add_argument("--ckpt", default=None, help="tujia-ctc-v0.2.pt")
    ap.add_argument("--encoder", default=None, help="tujia-xlsr53-ft directory")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--mode", choices=["none", "qwen3", "moss", "whisper"], default=None,
                    help="optional second channel")
    ap.add_argument("--seg", type=float, default=10.0, help="window target length (s)")
    ap.add_argument("--search", type=float, default=2.0, help="silence search radius (s)")
    ap.add_argument("--blank-bias", type=float, default=None,
                    help="压低 CTC blank 的 logit（0 关闭；1.0 少丢字、SER 基本不变；"
                         "1.5 更偏「宁错不删」）")
    ap.add_argument("--zipa-root", default=None, help="local ZIPA checkout (zipa backend)")
    ap.add_argument("--zipa-python", default=None, help="python of the ZIPA environment")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.frontend:
        settings.frontend = args.frontend
    if args.ckpt:
        settings.ckpt = args.ckpt
    if args.encoder:
        settings.encoder = args.encoder
    if args.device:
        settings.device = args.device
    if args.blank_bias is not None:
        settings.blank_bias = args.blank_bias
    if args.zipa_root:
        settings.audio_root = args.zipa_root
    if args.zipa_python:
        settings.audio_py = args.zipa_python
    if args.no_default_dict:
        settings.default_dict = False

    if not Path(args.wav).expanduser().is_file():
        print(f"错误：找不到音频：{args.wav}", file=sys.stderr)
        return 2

    try:
        res = recognize(
            args.wav,
            dict_path=args.dicts or None,
            settings=settings,
            mode=args.mode,
            seg=args.seg,
            search=args.search,
            tone=args.tone,
            device=args.device,
        )
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 3
    except Exception as e:  # ZIPA errors are runtime failures, not bugs to re-raise
        from .zipa import ZipaError

        if isinstance(e, ZipaError):
            print(f"ZIPA 失败：{e}", file=sys.stderr)
            return 4
        raise

    if not args.quiet:
        _print_report(res, args.format)

    target = args.json if args.json is not None else args.out
    if target is not None:
        payload = json.dumps(res, ensure_ascii=False, indent=1)
        if target == "-":
            print(payload)
        else:
            path = Path(target).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
            if not args.quiet:
                print(f"-> {path}")
    return 0
