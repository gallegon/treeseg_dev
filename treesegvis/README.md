# treesegvis

This is the visualizer for output from the Tree Segmentationa algorithm intermediate steps.

Implemented using p5js, which is fetched by the browser at runtime (for now).

All input images must be encoded in the proper format:
```
R = (ID >> 16) % 256
G = (ID >> 8) % 256
B = ID & 256
```