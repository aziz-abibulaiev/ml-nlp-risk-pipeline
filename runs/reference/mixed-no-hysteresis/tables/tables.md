# Run reference/mixed-no-hysteresis

Events: 26025 (failed requests: 0)

## Operating points

| operating_point   |   TP |   FP |   FN |    TN |   Precision |   Recall |     F1 |    FPR | Recall_CI95      | FPR_CI95         |
|:------------------|-----:|-----:|-----:|------:|------------:|---------:|-------:|-------:|:-----------------|:-----------------|
| Medium+           | 1025 |  292 |    0 | 24708 |      0.7783 |   1.0000 | 0.8753 | 0.0117 | [0.9964, 1.0000] | [0.0104, 0.0131] |
| Low+              | 1025 | 5447 |    0 | 19553 |      0.1584 |   1.0000 | 0.2734 | 0.2179 | [0.9964, 1.0000] | [0.2128, 0.2230] |

## Operating points, distinct requests (each distinct method+endpoint+payload+headers once)

| operating_point   |   distinct_attacks |   distinct_benign |   TP |   FP |   FN |    TN |   Recall |    FPR | Recall_CI95      | FPR_CI95         |
|:------------------|-------------------:|------------------:|-----:|-----:|-----:|------:|---------:|-------:|:-----------------|:-----------------|
| Medium+           |                197 |             25000 |  197 |  292 |    0 | 24708 |   1.0000 | 0.0117 | [0.9814, 1.0000] | [0.0104, 0.0131] |
| Low+              |                197 |             25000 |  197 | 5447 |    0 | 19553 |   1.0000 | 0.2179 | [0.9814, 1.0000] | [0.2128, 0.2230] |

## Risk levels by class

| class   |   normal |   low |   medium |   high |   critical |   total |
|:--------|---------:|------:|---------:|-------:|-----------:|--------:|
| attack  |        0 |     0 |      110 |    905 |         10 |    1025 |
| benign  |    19553 |  5155 |       81 |    207 |          4 |   25000 |

## Risk levels by attack group

| attack_group        |   normal |   low |   medium |   high |   critical |   total |
|:--------------------|---------:|------:|---------:|-------:|-----------:|--------:|
| api_key_exposure    |        0 |     0 |        0 |     30 |          0 |      30 |
| auth_bruteforce     |        0 |     0 |        0 |     80 |          0 |      80 |
| business_logic      |        0 |     0 |        0 |     30 |          0 |      30 |
| cmd_injection       |        0 |     0 |        0 |     50 |          0 |      50 |
| credential_stuffing |        0 |     0 |        0 |     50 |          0 |      50 |
| data_exposure       |        0 |     0 |        5 |     45 |          0 |      50 |
| dos                 |        0 |     0 |       15 |     75 |          0 |      90 |
| idor                |        0 |     0 |       20 |     20 |          0 |      40 |
| ldap_injection      |        0 |     0 |        0 |     40 |          0 |      40 |
| misconfiguration    |        0 |     0 |       10 |     30 |          0 |      40 |
| nosql_injection     |        0 |     0 |        0 |    100 |         10 |     110 |
| sql_injection       |        0 |     0 |        0 |    225 |          0 |     225 |
| ssrf                |        0 |     0 |       10 |     80 |          0 |      90 |
| ssrf_sensitive      |        0 |     0 |        0 |     10 |          0 |      10 |
| websocket           |        0 |     0 |        0 |     20 |          0 |      20 |
| xss                 |        0 |     0 |       50 |     20 |          0 |      70 |
