#!/usr/bin/env bash
# Grade, add grain/vignette, fade, mux the soundtrack and encode the final MP4.
# Usage: scripts/encode.sh <frames_dir> <audio.wav> <out.mp4> [fps]
set -euo pipefail
FRAMES="$1"; AUDIO="$2"; OUT="$3"; FPS="${4:-24}"
N=$(ls "$FRAMES"/f_*.png | wc -l)
DUR=$(python3 -c "print($N/$FPS)")
FADE_OUT=$(python3 -c "print(max(0.0, $N/$FPS - 0.9))")
VF="scale=1920:804:flags=lanczos,unsharp=5:5:0.45:5:5:0.0,"
VF+="eq=contrast=1.05:saturation=1.06:gamma=0.98,"
VF+="colorbalance=rs=-0.03:gs=-0.01:bs=0.04:rh=0.035:gh=0.01:bh=-0.035,"
VF+="vignette=angle=PI/5:mode=forward,"
VF+="noise=c0s=5:c0f=t+u:c1s=2:c1f=t+u:c2s=2:c2f=t+u,"
VF+="fade=t=in:st=0:d=0.5,fade=t=out:st=${FADE_OUT}:d=0.9,format=yuv420p"
ffmpeg -y -hide_banner -loglevel warning \
  -framerate "$FPS" -start_number 1 -i "$FRAMES/f_%04d.png" -i "$AUDIO" \
  -vf "$VF" -c:v libx264 -preset slow -crf 20 -tune film -pix_fmt yuv420p \
  -c:a aac -b:a 192k -af "afade=t=out:st=${FADE_OUT}:d=0.9" \
  -t "$DUR" -movflags +faststart "$OUT"
echo "wrote $OUT ($N frames, ${DUR}s)"
