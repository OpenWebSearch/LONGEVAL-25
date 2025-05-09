# PyTerrier Baseline for LongEval'25

This directory contains a PyTerrier baseline for the retrieval task of [LongEval 2025](https://clef-longeval.github.io/). This baseline uses the [LongEval ir_datasets extension](https://github.com/clef-longeval/ir-datasets-longeval) without modification and tracks resource consumption for indexing and retrieval in the ir_metadata format.

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

To create a run, please run:

```
./baseline.py --dataset longeval-sci/clef-2025-test --output output --index indexes
```

## Verify that your outputs are valid

To verify that your submission in the `output` directory is valid, please run:

```
tira-cli upload --dry-run --directory output --dataset sci-spot-check-with-prior-data-20250322-training
```
If your `output` directory is valid, it should report something like:

![Screenshot_20250428_133016](https://github.com/user-attachments/assets/50a4ee85-e599-4c2f-bb83-dcb382d889f4)


## Submissions

```
./baseline.py --dataset longeval-sci/spot-check/with-prior-data --output output --index indexes
```

## Optional: Code Submission to TIRA

As optional alternative to run submissions, you can make code submissions where the tira client will build a docker image of your approach from the source code and upload the image to TIRA.io so that your software can run in TIRA.io. To submit this baseline as code submission to TIRA, please run (more detailed information are available in the [documentation](https://docs.tira.io/participants/participate.html#submitting-your-submission):

```
tira-cli code-submission --dry-run --path . --task longeval-2025 --dataset sci-spot-check-with-prior-data-20250322-training --command '/baseline.py --dataset $inputDataset --index /tmp/indexes --output $outputDir'
```

If this is successfull, please re-run with removed the `--dry-run` flag to upload the software to TIRA.
