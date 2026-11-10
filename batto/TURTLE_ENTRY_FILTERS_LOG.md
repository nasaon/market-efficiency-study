터틀 진입 필터 실험 — RSI·스토캐스틱·MACD·일목 16조합 (청산=채택안 고정)
바스켓 9(코인3+미국강추세6)·robust: Train 4국면 중앙값Calmar 평균 / Test 2국면 / 종목승률

  조합(추가 필터)                       Train   Test    종목승    평균거래수
  무필터(기준=현재)                       1.33   0.62    0/9      41.9  (기준)
  RSI                              1.33   0.62    0/9      41.9
  Stoch                            1.33   0.62    0/9      41.9
  RSI+Stoch                        1.33   0.62    0/9      41.9
  MACD                             1.27   0.62    3/9      41.1
  RSI+MACD                         1.27   0.62    3/9      41.1
  Stoch+MACD                       1.27   0.62    3/9      41.1
  RSI+Stoch+MACD                   1.27   0.62    3/9      41.1
  Ichi                             1.01   0.56    1/9      39.9
  RSI+Ichi                         1.01   0.56    1/9      39.9
  Stoch+Ichi                       1.01   0.56    1/9      39.9
  RSI+Stoch+Ichi                   1.01   0.56    1/9      39.9
  MACD+Ichi                        0.98   0.56    0/9      39.1
  RSI+MACD+Ichi                    0.98   0.56    0/9      39.1
  Stoch+MACD+Ichi                  0.98   0.56    0/9      39.1
  RSI+Stoch+MACD+Ichi              0.98   0.56    0/9      39.1

  기준(무필터): Train 1.33 · Test 0.62
  Train 최고 조합: RSI (Train 1.33·Test 0.62·종목 0/9)
  ※ 필터 켤수록 거래수↓(엄격). 종목승 5/9 안팎=동전던지기(robust 엣지 아님).
