# Патч: смешивать по маске HR и результат диффузии (а не LQ)

Идея (замечание преподавателя): вне маски известная область берётся из эталона **HR**, а не из входного LQ; так сравнивается работа модели, когда у неё есть только LR, и когда есть «куски» в хорошем качестве.

Строки с `-` удалить, с `+` добавить; остальной код ячеек не меняется. Нумерация по `SUPIR_experiments-9.ipynb`; в вашем серверном ноутбуке найдите те же строки по тексту. Проверено: синтаксис и логика (на заглушках, без реального SUPIR; сам SUPIR с этой правкой не запускался).

Новые методы в `results.csv`: `A_HR` (HR как известная область внутри сэмплера) и `C_HR` (полный SUPIR + склейка с HR). Прежние `A` и `C` (LQ) остаются для сравнения.


## Ячейка 24 (`masked_batchify_sample`, `masked_enhance`)

```diff
-                               keep_source="lq"):
+                               keep_source="lq", x_ref=None):
+        if x_ref is not None: z_ref = model.encode_first_stage(x_ref)   # известная область = эталон HR, а не LQ
-                       skip_empty=True, keep_source="lq"):
+                       skip_empty=True, keep_source="lq", ref=None):
-        if skip_empty and mask.max() == 0: return image.copy()          # маска «всё 0»: сеть не нужна
+        if ref is not None and ref.shape != image.shape: raise ValueError(f"ref {ref.shape} != изображение {image.shape}")
+        if skip_empty and mask.max() == 0: return (image if ref is None else ref).copy()   # маска «всё 0»: сеть не нужна
+        x_ref = None
+        if ref is not None:
+            x_ref = torch.from_numpy(np.asarray(Image.fromarray(ref).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(profile["device"])
-                                     keep_source=keep_source, tile=profile["sampler_tile"] if tiled else None,
+                                     keep_source=keep_source, x_ref=x_ref, tile=profile["sampler_tile"] if tiled else None,
-        if hard_composite: res = composite(res, image, mask, profile["mask_feather_sigma"] * 8)
+        if hard_composite: res = composite(res, image if ref is None else ref, mask, profile["mask_feather_sigma"] * 8)
```


## Ячейка 57 (эксперимент)

```diff
-def enhance(img, mask):
-    return fake_enhancer(img, mask) if USE_FAKE else masked_enhance(model, img, mask, prof, seed=SEED, steps=prof["edm_steps"], hard_composite=HARD_COMPOSITE)
+def enhance(img, mask, ref=None):
+    return fake_enhancer(img, mask) if USE_FAKE else masked_enhance(model, img, mask, prof, seed=SEED, steps=prof["edm_steps"], hard_composite=HARD_COMPOSITE, ref=ref)
+            t0 = time.time(); out_ah = full if mk == "ones" else enhance(lq_up, m, ref=hr); t_ah = t_full if mk == "ones" else time.time() - t0   # известная область = HR
-            for method, im, sec in (("LQ", lq_up, 0.0), ("A", out_a, t_a), ("C", out_c, t_full)):
+            out_ch = composite(full, hr, m, feather)                                                                  # склейка с HR
+            for method, im, sec in (("LQ", lq_up, 0.0), ("A", out_a, t_a), ("C", out_c, t_full), ("A_HR", out_ah, t_ah), ("C_HR", out_ch, t_full)):
```


## Ячейка 58 (графики)

```diff
-        piv[[c for c in ("LQ", "A", "C") if c in piv]].plot.bar(ax=ax, rot=30); ax.set_title(deg); ax.set_xlabel(""); ax.grid(axis="y", alpha=.3)
+        piv[[c for c in ("LQ", "A", "C", "A_HR", "C_HR") if c in piv]].plot.bar(ax=ax, rot=30); ax.set_title(deg); ax.set_xlabel(""); ax.grid(axis="y", alpha=.3)
```
