import gnn_tf
import os.path
import numpy as np

data_set_name = ["mutag", "proteins", "cni1", "dd"]
performance = {}
for name_file in data_set_name:
    gnn_tf.args.data = name_file
    gnn_tf.args.epoch = 100
    if name_file not in performance.keys():
        performance[name_file] = []
    for i in range(10):
        print("-" * 30, name_file, "test: ", i)
        performance[name_file].append(gnn_tf.main())
print()
for key, value in performance.items():
    with open(os.path.join("logs", key+".txt"), "a") as f:
        for v in value:
            f.write(str(v) + "\n")
    print("%s: average acc is %f, variance is %f." % (key, np.average(value), np.std(value)))