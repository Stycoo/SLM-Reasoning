off_policy_candidate_rule_scores = [1, 1, 0]
off_policy_candidate_orm_scores = [0.5, 0.7, 0.9]
off_policy_candidates = ['a', 'b', 'c']
 
stats = list(zip(off_policy_candidate_rule_scores, off_policy_candidate_orm_scores, off_policy_candidates))
off_policy_chosen = max(stats, key=lambda x: (x[0], x[1]))[2]
off_policy_rejected = min(stats, key=lambda x: (x[0], x[1]))[2]

print(off_policy_chosen, off_policy_rejected)