stats_names = ["AP .5", "AP .75", "AP .95",
               "AR .5", "AR .75", "AR .95",
               "F1 .5", "F1 .75", "F1 .95"]
scores = [0.1, 0, 0,
          0, 0.1, 0,
          0, 0, 0.1]

info_str = list(zip(stats_names, scores))

print(info_str)