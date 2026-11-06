# Contributing

Use a new branch and keep generated runs outside tracked source files. Describe behavior changes and include validation appropriate to the changed code. Report the input dimensions, actual primitive count, main steps, optimizer updates, renderer build and metric domain when discussing reconstruction quality.

For codec changes, test fresh-process decoding, complete-file byte accounting and decoded reconstruction quality. Preserve upstream licenses and do not replace an original implementation without updating stream compatibility checks. A checkpoint is not a compressed bitstream.

Before submitting changes:

```bash
python -m pytest -q tests
```

Do not add datasets, images, trained weights, compressed streams, experiment outputs, credentials or compiled extensions to Git.
