#!/usr/bin/env bash
# Burn one language's captions into the demo recording (hardsub, always visible to graders).
# The tracks are separate: captions.es.srt (Spanish) and captions.en.srt (English) — never both
# on screen at once.
#
# Usage: ./burn_captions.sh input.mp4 [es|en|path.srt] [output.mp4]
#   ./burn_captions.sh demo.mp4            # Spanish (default) -> demo_es.mp4
#   ./burn_captions.sh demo.mp4 en         # English           -> demo_en.mp4
#   ./burn_captions.sh demo.mp4 captions.es.srt out.mp4
set -euo pipefail

IN="${1:?usage: burn_captions.sh input.mp4 [es|en|path.srt] [output.mp4]}"
LANG_ARG="${2:-es}"

case "$LANG_ARG" in
  es) SRT="captions.es.srt"; SUF="_es" ;;
  en) SRT="captions.en.srt"; SUF="_en" ;;
  *)  SRT="$LANG_ARG";       SUF="_captioned" ;;   # explicit .srt path
esac
OUT="${3:-${IN%.*}${SUF}.mp4}"

# White text, black outline, bottom-centred, comfortable margin. DejaVu Sans has the
# arrows (-> <->) and accents; swap FontName if your box lacks it.
STYLE="FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=40,Alignment=2"

ffmpeg -i "$IN" \
  -vf "subtitles=${SRT}:force_style='${STYLE}'" \
  -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  -c:a copy \
  "$OUT"

echo "wrote $OUT (subs: $SRT)"

# --- alternatives ---------------------------------------------------------------
# Both languages as SELECTABLE (toggleable) soft-sub tracks in one .mp4 — the viewer
# picks one at a time, so they are still never shown simultaneously:
#   ffmpeg -i in.mp4 -i captions.es.srt -i captions.en.srt \
#     -map 0 -map 1 -map 2 -c copy -c:s mov_text \
#     -metadata:s:s:0 language=spa -metadata:s:s:1 language=eng out.mp4
# Preview a track before committing:
#   mpv --sub-file=captions.es.srt in.mp4
