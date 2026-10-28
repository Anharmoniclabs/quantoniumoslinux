# JC V0.6 creator-owned eight-pose action cels

These 8 lossless WebP files are pixel-identical conversions of JC's eight kick poses from page 3 of the creator's *Heralds of the Cemi — Full Cast Movement and In-Betweens Bible*. They are included publicly in the repository by the creator's explicit request. Copyright is not waived.

The original PDF and 43 MB full-cast Bible are not included here. The supplied preview MP4 is a 48-frame / 24 fps pose-based animation proof.

Use `bash runtime/experiments/quantonium_video/fetch_and_render_v06.sh` after fetching the video branch. The runner verifies SHA-256 sums, converts lossless WebP frames to RGBA PNGs outside the repo, and renders with your existing V0.4 5,000-step motion checkpoint.

This is not Blender rigging, learned articulation, or generative video: the poses were drawn previously, and motion between them is timed cel selection with small learned root translation.
