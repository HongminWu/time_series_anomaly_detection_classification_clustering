time_series_anomaly_detection_classification_clustering
TRAIN_FILES = ['/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/arabic/', # 0
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/CK/', # 1
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/character/', # 2
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/Action3D/', # 3
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/Activity/', # 4
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/arabic_voice/', # 5
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/JapaneseVowels/', # 6

               # New benchmark datasets
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/AREM/', # 7
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/gesture_phase/', # 8
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/HT_Sensor/', # 9
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/MovementAAL/', # 10
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/HAR/', # 11
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/occupancy_detect/', # 12
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/eeg/',  # 13
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/ozone/', # 14
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/daily_sport/',  # 15
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/eeg2/',  # 16
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/MHEALTH/',  # 17
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp2_data1a/',  # 18
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp2_data1b/',  # 19
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp2_data3/',  # 20
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp2_data4/',  # 21
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp3_data2/',  # 22
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp3_data1/',  # 23
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/uwave/',  # 24
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/opportunity/',  # 25
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/pamap2/',  # 26
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/ArabicDigits/',  # 27
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/AUSLAN/',  # 28
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/CharacterTrajectories/',  # 29
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/CMUsubject16/',  # 30
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/ECG/',  # 31
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/JapaneseVowels/',  # 32
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/KickvsPunch/',  # 33
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/Libras/',  # 34
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/NetFlow/',  # 35
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/PEMS/',  # 36
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/UWave/',  # 37
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/Wafer/',  # 38
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/WalkvsRun/',  # 39
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/digitshape_random/',  # 40
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp1/',  # 41
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp2/',  # 42
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp3/',  # 43
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp4/',  # 44
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp5/',  # 45
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/pendigits/',  # 46
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/shapes_random/',  # 47

               # customize dataset by HongminWu
               '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/baxter_pnp_anomalies/', # 48
               ]

TEST_FILES = ['/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/arabic/', # 0
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/CK/', # 1
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/character/', # 2
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/Action3D/', # 3
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/Activity/', # 4
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/arabic_voice/', # 5
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/JapaneseVowels/', # 6

              # New benchmark datasets
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/AREM/', # 7
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/gesture_phase/', # 8
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/HT_Sensor/',  # 9
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/MovementAAL/',  # 10
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/HAR/',  # 11
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/occupancy_detect/',  # 12
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/eeg/', # 13
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/ozone/',  # 14
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/daily_sport/',  # 15
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/eeg2/',  # 16
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/MHEALTH/',  # 17
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp2_data1a/',  # 18
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp2_data1b/',  # 19
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp2_data3/',  # 20
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp2_data4/',  # 21
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp3_data2/',  # 22
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/EEG_Comp3_data1/',  # 23
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/uwave/',  # 24
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/opportunity/',  # 25
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/pamap2/',  # 26
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/ArabicDigits/',  # 27
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/AUSLAN/',  # 28
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/CharacterTrajectories/',  # 29
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/CMUsubject16/',  # 30
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/ECG/',  # 31
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/JapaneseVowels/',  # 32
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/KickvsPunch/',  # 33
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/Libras/',  # 34
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/NetFlow/',  # 35
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/PEMS/',  # 36
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/UWave/',  # 37
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/Wafer/',  # 38
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/WalkvsRun/',  # 39
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/digitshape_random/',  # 40
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp1/',  # 41
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp2/',  # 42
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp3/',  # 43
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp4/',  # 44
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/lp5/',  # 45
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/pendigits/',  # 46
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/UCI_DATASETS/shapes_random/',  # 47
               # customize dataset by HongminWu
              '/home/birl_wu/time_series_anomaly_detection_classification_clustering/dataset/baxter_pnp_anomalies/', # 48              
              ]

MAX_NB_VARIABLES = [13,  # 0
                    136,  # 1
                    30,  # 2
                    570,  # 3
                    570,  # 4
                    39,  # 5
                    12,  # 6

                    # New benchmark datasets
                    7,  # 7
                    18,  # 8
                    11,  # 9
                    4,  # 10
                    9,  # 11
                    5,  # 12
                    13,  # 13
                    72,  # 14
                    45,  #15
                    64,  #16
                    23,  #17
                    6,  #18
                    7,  #19
                    3,  #20
                    28,  #21
                    64,  #22
                    64,  #23
                    3,  #24
                    77,  #25
                    52,  #26
                    13,  #27
                    22,  #28
                    3,  #29
                    62,  #30
                    2,  #31
                    12,  #32
                    62,  #33
                    2,  #34
                    4,  #35
                    963,  #36
                    3,  #37
                    6,  #38
                    62,  #39
                    2,  #40
                    6,  #41
                    6,  # 42
                    6,  # 43
                    6,  # 44
                    6,  # 45
                    2, #46
                    2, #47

                    # customize dataset by HongminWu
                    12, # 48
                    
                    ]

MAX_TIMESTEPS_LIST = [93,  # 0
                      71,  # 1
                      173,  # 2
                      100, # 3
                      337, # 4
                      91, # 5
                      26, # 6

                      # New benchmark datasets
                      480, # 7
                      214, # 8
                      5396, # 9
                      119, # 10
                      128, # 11
                      3758, # 12
                      117, # 13
                      291, # 14
                      125, #15
                      256, #16
                      42701, #17
                      896, #18
                      1152, #19
                      1152, #20
                      500,#21
                      7794, #22
                      3000, #23
                      315, #24
                      24, #25
                      34, #26
                      93, #27
                      96, #28
                      205, #29
                      534, #30
                      147, #31
                      26, #32
                      761, #33
                      45, #34
                      994, #35
                      144, #36
                      315, #37
                      198, #38
                      1918, #39
                      97, #40
                      15, #41
                      15, #42
                      15, #43
                      15, #44
                      15, #45
                      8, #46
                      97, #47

                      # customize dataset by HongminWu
                      40, # 48                      

                      ]


NB_CLASSES_LIST = [10, # 0
                   7, # 1
                   20, # 2
                   20, # 3
                   16, # 4
                   88, # 5
                   9, # 6

                   # New benchmark datasets
                   7, # 7
                   5, # 8
                   3, # 9
                   2, # 10
                   6, # 11
                   2, # 12
                   2, # 13
                   2, # 14
                   19, #15
                   2, #16
                   13, #17
                   2, #18
                   2, #19
                   2, #20
                   2,#21
                   29, #22
                   2, #23
                   8, #24
                   18, #25
                   12, #26
                   10, #27
                   95, #28
                   20, #29
                   2, #30
                   2, #31
                   9, #32
                   2, #33
                   15, #34
                   2, #35
                   7, #36
                   8, #37
                   2, #38
                   2, #39
                   4, #40
                   4, #41
                   5, #42
                   4, #43
                   3, #44
                   5, #45
                   10, #46
                   3,#47

                   # customize dataset by HongminWu
                   4, # 48                     

                   ]

NB_CLASSES_NAMES = [10, # 0
                   7, # 1
                   20, # 2
                   20, # 3
                   16, # 4
                   88, # 5
                   9, # 6

                   # New benchmark datasets
                   7, # 7
                   5, # 8
                   3, # 9
                   2, # 10
                   6, # 11
                   2, # 12
                   2, # 13
                   2, # 14
                   19, #15
                   2, #16
                   13, #17
                   2, #18
                   2, #19
                   2, #20
                   2,#21
                   29, #22
                   2, #23
                   8, #24
                   18, #25
                   12, #26
                   10, #27
                   95, #28
                   20, #29
                   2, #30
                   2, #31
                   9, #32
                   2, #33
                   15, #34
                   2, #35
                   7, #36
                   8, #37
                   2, #38
                   2, #39
                   4, #40
                   4, #41
                   5, #42
                   4, #43
                   3, #44
                   5, #45
                   10, #46
                   3,#47

                   # customize dataset by HongminWu
                   ['OBJECT_SLIPPING', 'TOOL_COLLISION', 'HUMAN_COLLISION', 'NO_OBJECT'], # 48                     

                   ]    
