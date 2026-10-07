# Case study: re2-tt full, DELAY faults

Source: `artifacts/formal-rerun-20260930/main-20260930/merit_full/re2-tt/results/diagnostics/<case>.json`.
w = `metric_service_evidence` (w_j), beta = `service_banzhaf` (beta_i, unnormalized), phi_ij = `player_target_contributions[i][j]`; check: sum_j phi_ij == beta_i.

| case | MERIT root rank | root rank by w | argmax w |
|---|---:|---:|---|
| ts-auth-service_delay_1 | 1 | 1 | ts-auth-service |
| ts-auth-service_delay_2 | 37 | 37 | ts-food-map-mongo |
| ts-auth-service_delay_3 | 18 | 19 | ts-consign-price-service |
| ts-order-service_delay_1 | 1 | 2 | ts-seat-service |
| ts-order-service_delay_2 | 1 | 1 | ts-order-service |
| ts-order-service_delay_3 | 1 | 1 | ts-order-service |
| ts-route-service_delay_1 | 1 | 1 | ts-route-service |
| ts-route-service_delay_2 | 1 | 1 | ts-route-service |
| ts-route-service_delay_3 | 1 | 1 | ts-route-service |
| ts-train-service_delay_1 | 1 | 1 | ts-train-service |
| ts-train-service_delay_2 | 1 | 1 | ts-train-service |
| ts-train-service_delay_3 | 1 | 1 | ts-train-service |
| ts-travel-service_delay_1 | 1 | 4 | ts-train-service |
| ts-travel-service_delay_2 | 1 | 4 | ts-train-service |
| ts-travel-service_delay_3 | 12 | 7 | ts-train-service |

## ts-order-service_delay_1

Source file: `artifacts/formal-rerun-20260930/main-20260930/merit_full/re2-tt/results/diagnostics/ts-order-service_delay_1.json`. MERIT root rank 1; root rank by w 2.

| service | role | w | beta |
|---|---|---:|---:|
| ts-order-service | root, MERIT #1 | 0.4792 | 0.4660 |
| ts-travel-service | MERIT #2 | 0.0932 | 0.4150 |
| ts-preserve-service | MERIT #3 | 0.0926 | 0.3521 |
| ts-seat-service | argmax w | 1.0000 | 0.1379 |

Top-4 phi_ij for root ts-order-service (sum over all j = beta = 0.4660):

| target j | phi_ij | w_j |
|---|---:|---:|
| ts-seat-service | 0.2529 | 1.0000 |
| ts-order-service | 0.2024 | 0.4792 |
| ts-preserve-service | 0.0107 | 0.0926 |
| ts-admin-basic-info-service | 0.0000 | 0.0736 |

## ts-travel-service_delay_1

Source file: `artifacts/formal-rerun-20260930/main-20260930/merit_full/re2-tt/results/diagnostics/ts-travel-service_delay_1.json`. MERIT root rank 1; root rank by w 4.

| service | role | w | beta |
|---|---|---:|---:|
| ts-travel-service | root, MERIT #1 | 0.2681 | 0.3854 |
| ts-basic-service | MERIT #2 | 0.0097 | 0.3369 |
| ts-ui-dashboard | MERIT #3 | 0.3096 | 0.3096 |
| ts-train-service | argmax w | 1.0000 | 0.2289 |

Top-4 phi_ij for root ts-travel-service (sum over all j = beta = 0.3854):

| target j | phi_ij | w_j |
|---|---:|---:|
| ts-train-service | 0.1848 | 1.0000 |
| ts-route-service | 0.1006 | 0.5443 |
| ts-travel-service | 0.0614 | 0.2681 |
| ts-seat-service | 0.0385 | 0.1558 |

## ts-travel-service_delay_2

Source file: `artifacts/formal-rerun-20260930/main-20260930/merit_full/re2-tt/results/diagnostics/ts-travel-service_delay_2.json`. MERIT root rank 1; root rank by w 4.

| service | role | w | beta |
|---|---|---:|---:|
| ts-travel-service | root, MERIT #1 | 0.2582 | 0.4256 |
| ts-travel2-service | MERIT #2 | 0.0745 | 0.3580 |
| ts-basic-service | MERIT #3 | 0.0067 | 0.3415 |
| ts-train-service | argmax w | 1.0000 | 0.2291 |

Top-4 phi_ij for root ts-travel-service (sum over all j = beta = 0.4256):

| target j | phi_ij | w_j |
|---|---:|---:|
| ts-train-service | 0.1848 | 1.0000 |
| ts-route-service | 0.0987 | 0.5340 |
| ts-seat-service | 0.0759 | 0.3075 |
| ts-travel-service | 0.0661 | 0.2582 |

