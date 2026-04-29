/* Software fmaf: target lacks hardware FMA and SDK doesn't provide it. */
float fmaf(float x, float y, float z) { return x * y + z; }
