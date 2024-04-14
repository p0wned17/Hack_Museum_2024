def classify(response_list: list, score_threshold: float = 0.4):
    if response_list:
        group_scores = {}
        for resp in response_list:
            if resp["score"] > score_threshold:
                group = resp["group"]
                if group in group_scores:
                    group_scores[group]["count"] += 1
                    group_scores[group]["score_sum"] += resp["score"]
                else:
                    group_scores[group] = {"count": 1, "score_sum": resp["score"]}

        if group_scores:
            # Сортировка словаря по количеству и сумме score
            sorted_groups = sorted(
                group_scores.items(),
                key=lambda x: (x[1]["count"], x[1]["score_sum"]),
                reverse=True,
            )
            most_frequent_group = sorted_groups[0][0]
        else:
            most_frequent_group = response_list[0]["group"]

        result = {
            "object_id": response_list[0]["object_id"],
            "img_name": response_list[0]["img_name"],
            "group": most_frequent_group,
        }

        return result
