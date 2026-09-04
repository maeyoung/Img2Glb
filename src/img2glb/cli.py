"""img2glb 명령줄 인터페이스.

    img2glb --model trellis2 --image cat.png --output cat.glb
    img2glb preview cat.glb
    img2glb inspect cat.glb
"""
import argparse
import sys
from pathlib import Path

from . import __version__
from .backends import BACKENDS
from .bg import DEFAULT as BG_DEFAULT, REMOVERS

_SUBCOMMANDS = ("generate", "preview", "inspect", "backends", "remove-bg",
                "metadata")


def _build_parser():
    p = argparse.ArgumentParser(
        prog="img2glb",
        description="이미지 한 장에서 텍스처(PBR) 가 입혀진 GLB 를 생성합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예시:\n"
               "  img2glb --model trellis2 --image cat.png --output cat.glb\n"
               "  img2glb remove-bg photo.jpg -o photo_rgba.png   # 배경 제거만\n"
               "  img2glb preview cat.glb --resolution 1024\n"
               "  img2glb inspect cat.glb\n",
    )
    p.add_argument("--version", action="version", version=f"img2glb {__version__}")
    sub = p.add_subparsers(dest="command")

    # ---- generate (기본) ----
    g = sub.add_parser("generate", help="이미지 -> GLB 생성 (기본 명령)")
    g.add_argument("--model", default="trellis2", choices=sorted(BACKENDS),
                   help="사용할 백엔드")
    g.add_argument("--image", "-i", required=True, type=Path, help="입력 이미지")
    g.add_argument("--output", "-o", type=Path, default=None,
                   help="출력 GLB 경로 (기본: output/glb/<입력이름>.glb)")
    g.add_argument("--seed", type=int, default=2025)
    g.add_argument("--no-preview", dest="preview", action="store_false",
                   help="미리보기 PNG 생성을 건너뛴다")
    g.set_defaults(preview=True)
    g.add_argument("--no-remove-bg", dest="remove_bg", action="store_false",
                   help="배경 자동 제거를 끈다 (배경이 형상에 포함될 수 있음)")
    g.set_defaults(remove_bg=True)
    g.add_argument("--bg-method", default=BG_DEFAULT, choices=sorted(REMOVERS),
                   help=f"배경제거 방식 (기본 {BG_DEFAULT})")
    g.add_argument("--keep-rgba", action="store_true",
                   help="배경 제거 결과를 output/rgba/ 에 남긴다")
    g.add_argument("--description", "-d", default=None,
                   help="에셋 설명. 생략하면 사양 기반으로 자동 생성한다")
    g.add_argument("--tags", default=None,
                   help="추가 태그 (쉼표 구분). 자동 태그와 합쳐진다")
    g.add_argument("--no-metadata", dest="metadata", action="store_false",
                   help="메타데이터 JSON 을 만들지 않는다")
    g.set_defaults(metadata=True)
    g.add_argument("--verbose", "-v", action="store_true")
    for backend in BACKENDS.values():
        backend.add_arguments(g)

    # ---- preview ----
    pv = sub.add_parser("preview", help="GLB 를 여러 각도로 렌더")
    pv.add_argument("glb", type=Path)
    pv.add_argument("--output", "-o", type=Path, default=None,
                    help="출력 PNG (기본: GLB 옆에 <이름>_preview.png)")
    pv.add_argument("--resolution", type=int, default=768)
    pv.add_argument("--views", default="0,90,180,270", help="azimuth 목록")
    pv.add_argument("--elev", type=float, default=0.0)

    # ---- inspect ----
    ins = sub.add_parser("inspect", help="GLB 스펙 출력")
    ins.add_argument("glb", nargs="+", type=Path)

    # ---- remove-bg ----
    rb = sub.add_parser("remove-bg", help="이미지 배경 제거 (모델과 독립 동작)")
    rb.add_argument("image", type=Path)
    rb.add_argument("--output", "-o", type=Path, default=None,
                    help="출력 PNG (기본: 입력과 같은 위치에 _rgba.png)")
    rb.add_argument("--method", default=BG_DEFAULT, choices=sorted(REMOVERS))
    rb.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # ---- metadata ----
    md = sub.add_parser("metadata", help="기존 GLB 의 메타데이터 JSON 생성/갱신")
    md.add_argument("glb", nargs="+", type=Path)
    md.add_argument("--description", "-d", default=None)
    md.add_argument("--tags", default=None, help="추가 태그 (쉼표 구분)")
    md.add_argument("--source", type=Path, default=None, help="원본 이미지 경로")
    md.add_argument("--preview", type=Path, default=None, help="미리보기 PNG 경로")
    md.add_argument("--print", dest="do_print", action="store_true",
                    help="파일로 쓰지 않고 표준출력에 낸다")

    # ---- backends ----
    sub.add_parser("backends", help="사용 가능한 백엔드 / 배경제거 방식 목록")
    return p


def _split_tags(raw):
    return [t.strip() for t in raw.split(",") if t.strip()] if raw else []


def _default_output(image: Path) -> Path:
    """에셋마다 폴더를 하나씩 둔다: output/glb/<이름>/<이름>.glb

    GLB 와 메타데이터 JSON 이 한 폴더에 모이므로 에셋 단위로 옮기거나
    올리기 쉽다.
    """
    stem = image.stem
    return Path.cwd() / "output" / "glb" / stem / f"{stem}.glb"


def _default_render(glb: Path) -> Path:
    """미리보기는 GLB 와 같은 폴더에 둔다.

    에셋 하나가 한 폴더에 모여 있어야 통째로 옮기거나 올리기 쉽다.
    """
    return glb.parent / f"{glb.stem}_preview.png"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # 서브커맨드를 생략하면 generate 로 간주한다
    if argv and argv[0] not in _SUBCOMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv.insert(0, "generate")
    elif not argv:
        _build_parser().print_help()
        return 1

    args = _build_parser().parse_args(argv)

    if args.command == "backends":
        print("생성 백엔드:")
        for name, cls in sorted(BACKENDS.items()):
            print(f"  {name:12s} {cls.description}")
        print("\n배경제거 방식:")
        for name, cls in sorted(REMOVERS.items()):
            mark = " (기본)" if name == BG_DEFAULT else ""
            print(f"  {name:12s} {cls.description}{mark}")
            print(f"  {'':12s}   라이선스: {cls.license}")
        return 0

    if args.command == "remove-bg":
        from PIL import Image
        from .bg import has_alpha, remove_background
        img = Image.open(args.image)
        if has_alpha(img):
            print("[bg] 이미 알파 채널이 있습니다. 그대로 저장합니다.")
            out_img = img.convert("RGBA")
        else:
            print(f"[bg] {args.method} 로 배경 제거 중...")
            out_img = remove_background(img, method=args.method, device=args.device)
        out = args.output or args.image.with_name(f"{args.image.stem}_rgba.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out)
        print(f"[out] {out}  ({out_img.size[0]}x{out_img.size[1]} RGBA)")
        return 0

    if args.command == "inspect":
        from .tools.inspect import summarize
        for g in args.glb:
            summarize(g)
        return 0

    if args.command == "metadata":
        import json

        from .metadata import build, write
        for g in args.glb:
            if not g.is_file():
                print(f"파일이 없습니다: {g}", file=sys.stderr)
                return 2
            meta = build(g, source_image=args.source, description=args.description,
                         tags=_split_tags(args.tags), preview_image=args.preview)
            if args.do_print:
                print(json.dumps(meta, ensure_ascii=False, indent=2))
            else:
                print(f"[meta] {write(meta, g)}")
        return 0

    if args.command == "preview":
        from .tools.preview import render
        out = args.output or _default_render(args.glb)
        render(args.glb, out, resolution=args.resolution,
               views=[float(v) for v in args.views.split(",")], elev=args.elev)
        return 0

    # ---- generate ----
    if not args.image.is_file():
        print(f"입력 이미지를 찾을 수 없습니다: {args.image}", file=sys.stderr)
        return 2

    # 알파 채널이 없으면 배경을 자동으로 제거한다.
    # (TRELLIS.2 내장 rembg 는 비상업 라이선스라 쓰지 않는다)
    image_path = args.image
    _tmp_rgba = None
    from PIL import Image
    from .bg import has_alpha

    src = Image.open(image_path)
    if has_alpha(src):
        print("[bg] 알파 채널이 있어 배경 제거를 건너뜁니다.")
    elif not args.remove_bg:
        print("[bg] 배경 제거가 꺼져 있습니다. 배경이 형상에 포함될 수 있습니다.")
    else:
        from .bg import remove_background
        print(f"[bg] 알파 채널이 없어 {args.bg_method} 로 배경을 제거합니다...")
        rgba = remove_background(src, method=args.bg_method)
        if args.keep_rgba:
            image_path = Path.cwd() / "output" / "rgba" / f"{args.image.stem}_rgba.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            rgba.save(image_path)
            print(f"[bg] 저장: {image_path}")
        else:
            import tempfile
            _tmp_rgba = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            rgba.save(_tmp_rgba.name)
            _tmp_rgba.close()
            image_path = Path(_tmp_rgba.name)

    output = args.output or _default_output(args.image)
    backend = BACKENDS[args.model]()
    try:
        result = backend.generate(image_path, output, args)
    finally:
        if _tmp_rgba is not None:
            Path(_tmp_rgba.name).unlink(missing_ok=True)
    size = result.stat().st_size / 1e6
    print(f"[out] {result}  ({size:.2f} MB)")

    preview_path = None
    if args.preview:
        from .tools.preview import render
        preview_path = render(result, _default_render(result))

    if args.metadata:
        from .metadata import build, write
        gen = dict(getattr(backend, "stats", {}) or {})
        gen.setdefault("backend", args.model)
        meta = build(result, source_image=args.image, generation=gen,
                     description=args.description, tags=_split_tags(args.tags),
                     preview_image=preview_path)
        print(f"[meta] {write(meta, result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
