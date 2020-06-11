def dp(scores, labels):
    time_step = scores.shape[0]
    log_scores = np.log(softmax(scores, axis=-1))
    labels = np.append(np.array(labels), blank)
    # e.g. [1748 161 1748 24 1748 106 1748 1595 1748 1640 1748 0 1748]
    labels = [k for k,g in groupby(labels)]
    labels_len = len(labels)

    max_scores = np.full([time_step, labels_len], np.NINF, dtype=np.float32)
    path_record = np.full([time_step, labels_len], -1, dtype=np.int32)

    # init
    max_scores[0, 0] = log_scores[0, labels[0]]
    max_scores[0, 1] = log_scores[0, labels[1]]

    path_record[0, 0] = 0
    path_record[0, 1] = 1

    path = []

    # iter
    for t in range(1, time_step):
        for l in range(0, labels_len):
            index = 0
            if l == 0:
                max_scores[t, l] = max_scores[t-1, l] + log_scores[t, labels[l]]
            elif l == 1:
                max_scores[t, l] = max(max_scores[t-1, l], max_scores[t-1, l-1]) + log_scores[t, labels[l]]
                index = np.argmax([max_scores[t-1, l], max_scores[t-1, l-1]])
            else:
                if labels[l] != labels[l-2]:
                   max_scores[t, l] = max(max_scores[t-1, l], max_scores[t-1, l-1], max_scores[t-1, l-2]) + log_scores[t, labels[l]]
                   index = np.argmax([max_scores[t-1, l], max_scores[t-1, l-1], max_scores[t-1, l-2]])
                else:
                   max_scores[t, l] = max(max_scores[t-1, l], max_scores[t-1, l-1]) + log_scores[t, labels[l]]
                   index = np.argmax([max_scores[t-1, l], max_scores[t-1, l-1]])
            path_record[t, l] = l - index

    # back track
    max_scores[-1, :-2] = np.NINF
    index = np.argmax(max_scores[-1])
    prev = path_record[time_step-1, index]
    path.append(index)
    path.append(prev)
    for t in reversed(range(1, time_step-1)):
        prev = path_record[t, prev]
        path.append(prev)

    path.reverse()
    final_seq = np.take(labels, path)
    return final_seq