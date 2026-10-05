# Architecture

## Offline training
1. Validate and order transaction history chronologically.
2. Build behavioral features without using future observations.
3. Fit a supervised fraud classifier on labels.
4. Fit Isolation Forest primarily on legitimate transactions.
5. Store model parameters, threshold and feature contract.

## Online / streaming-style path
For each incoming transaction:
1. validate schema;
2. derive the same feature representation;
3. compute supervised fraud probability;
4. compute anomaly score;
5. combine both into a calibrated operational risk score;
6. compare with an alert threshold;
7. attach reason codes and send to an analyst queue.

## Why hybrid detection?
A supervised model learns known fraud patterns but depends on historical labels. An unsupervised detector can flag statistically unusual behaviour that may not resemble previously labelled fraud. Their combination creates a stronger monitoring baseline than either signal alone.

## Production notes
A real deployment should replace the stateless demo with a feature store or streaming state backend, model registry, schema validation, monitoring, encryption, access controls and audit logging.
