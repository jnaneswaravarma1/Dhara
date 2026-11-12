# Retrieval evaluation (50 questions, 10 out-of-scope)

Database-path index: 33,653 live CENTRAL sections. Production limit = 3 per retriever.

| Config | hit@1 | hit@3 | hit@5 | hit@5 95% CI | MRR@5 | context hit (keyword corpus + top 3) | answered out-of-scope |
|---|---|---|---|---|---|---|---|
| keyword_corpus_only | 5/50 | 6/50 | 6/50 | 0.04-0.22 | 0.11 | 6/50 | 0/10 |
| production_approx | 6/50 | 13/50 | 16/50 | 0.20-0.46 | 0.194 | 16/50 | 4/10 |
| bm25 | 9/50 | 17/50 | 19/50 | 0.26-0.52 | 0.259 | 21/50 | 10/10 |
| dense_lsa | 0/50 | 2/50 | 4/50 | 0.02-0.16 | 0.027 | 7/50 | 10/10 |
| rrf_bm25_dense | 3/50 | 9/50 | 10/50 | 0.10-0.32 | 0.115 | 13/50 | 10/10 |
| rrf_prod_dense | 4/50 | 8/50 | 10/50 | 0.10-0.32 | 0.122 | 13/50 | 10/10 |
| rrf_prod_bm25 | 9/50 | 17/50 | 17/50 | 0.22-0.48 | 0.253 | 21/50 | 10/10 |
| hybrid_all | 6/50 | 11/50 | 14/50 | 0.16-0.40 | 0.185 | 15/50 | 10/10 |
| hybrid_all_gated | 5/50 | 10/50 | 13/50 | 0.14-0.38 | 0.162 | 14/50 | 9/10 |
| rrf_prod_bm25_abstain | 9/50 | 16/50 | 16/50 | 0.20-0.46 | 0.243 | 20/50 | 4/10 |

Paired difference vs production_approx, mean (95% bootstrap interval). An interval that includes 0 means the gap could be noise:

| Config | hit@3 diff | context-hit diff |
|---|---|---|
| bm25 | +0.08 (-0.04 to +0.20) | +0.10 (-0.02 to +0.22) |
| rrf_prod_bm25_abstain | +0.06 (-0.04 to +0.16) | +0.08 (+0.00 to +0.18) |
| dense_lsa | -0.22 (-0.34 to -0.10) | -0.18 (-0.28 to -0.08) |
| rrf_bm25_dense | -0.08 (-0.20 to +0.04) | -0.06 (-0.16 to +0.04) |
| rrf_prod_dense | -0.10 (-0.20 to +0.00) | -0.06 (-0.14 to +0.02) |
| rrf_prod_bm25 | +0.08 (+0.00 to +0.18) | +0.10 (+0.02 to +0.20) |
| hybrid_all | -0.04 (-0.14 to +0.06) | -0.02 (-0.10 to +0.06) |
| hybrid_all_gated | -0.06 (-0.16 to +0.02) | -0.04 (-0.12 to +0.04) |

Abstain gate (refuse when the best BM25 score is below the cutoff; used cutoff 14.0, hand-set from this same data, so optimistic):

| Cutoff | answerable questions refused | out-of-scope refused |
|---|---|---|
| 10.0 | 0/50 | 2/10 |
| 12.0 | 2/50 | 4/10 |
| 14.0 | 7/50 | 6/10 |
| 16.0 | 14/50 | 8/10 |
| 18.0 | 22/50 | 9/10 |
| 20.0 | 29/50 | 9/10 |

## Questions the answer model would see NO expected section for (production_approx + keyword corpus)

- q01: Someone stole my phone from my bag while I was travelling in a bus (expected BNS:303, BNS:305; hybrid_all_gated rank: None)
- q02: A man on a bike grabbed my gold chain from my neck and rode away (expected BNS:304; hybrid_all_gated rank: None)
- q03: Two men threatened me with a knife and took my wallet and watch (expected BNS:309; hybrid_all_gated rank: None)
- q04: A man keeps following me and messaging me even after I told him to stop (expected BNS:78; hybrid_all_gated rank: None)
- q05: My manager makes unwelcome comments about my body and asks for favours at the office (expected BNS:75, POSH:9, POSH:4; hybrid_all_gated rank: None)
- q06: My husband and his parents keep harassing me for more dowry (expected BNS:85, DOWRY:4, DOWRY:3; hybrid_all_gated rank: None)
- q07: My sister died within a few years of marriage after being harassed for dowry (expected BNS:80; hybrid_all_gated rank: 4)
- q09: Someone is threatening to kill me over phone calls (expected BNS:351; hybrid_all_gated rank: None)
- q10: A shopkeeper took my money and gave me a fake product pretending it was genuine (expected BNS:318; hybrid_all_gated rank: None)
- q11: Someone posted false stories about me on social media to ruin my reputation (expected BNS:356; hybrid_all_gated rank: None)
- q12: My business partner took the money entrusted to him and used it for himself (expected BNS:316; hybrid_all_gated rank: None)
- q13: Local goons are demanding money from my shop and threaten to hurt my family if I refuse (expected BNS:308; hybrid_all_gated rank: None)
- q14: A speeding driver hit and killed a pedestrian (expected BNS:106, BNS:281; hybrid_all_gated rank: None)
- q16: Someone deliberately smashed the windows of my car (expected BNS:324; hybrid_all_gated rank: None)
- q17: A man was beaten badly with a stick and his arm was broken (expected BNS:117, BNS:115; hybrid_all_gated rank: None)
- q20: The police did not tell me why they arrested me (expected BNSS:47; hybrid_all_gated rank: None)
- q23: What is the penalty for riding a two wheeler without a helmet (expected MV:129, MV:194D; hybrid_all_gated rank: None)
- q25: Can I drive a car without having a driving licence (expected MV:3, MV:181; hybrid_all_gated rank: None)
- q28: I bought a washing machine that is defective and the company will not replace it (expected CPA:35, CPA:39; hybrid_all_gated rank: None)
- q29: How many years do I have to file a consumer case after a problem (expected CPA:69; hybrid_all_gated rank: None)
- q30: Which consumer forum should I approach for a claim of twenty lakh rupees (expected CPA:34; hybrid_all_gated rank: None)
- q33: What kind of information can the government refuse to give me (expected RTI:8; hybrid_all_gated rank: None)
- q34: My information request was rejected and I want to challenge that decision (expected RTI:19; hybrid_all_gated rank: None)
- q35: My husband beats me at home, can I get a court order to stop him (expected DV:18, DV:12; hybrid_all_gated rank: None)
- q38: A stranger used my name and password to pretend to be me online (expected IT:66C, IT:66D; hybrid_all_gated rank: None)
- q39: A fraudster pretended to be from my bank and took money from my account online (expected IT:66D, BNS:318; hybrid_all_gated rank: None)
- q40: Someone shared my private photos on the internet without my permission (expected IT:66E, IT:67; hybrid_all_gated rank: None)
- q41: My landlord wants me to leave the rented house but we never signed any written agreement (expected TPA:106; hybrid_all_gated rank: None)
- q43: The builder has delayed giving me possession of my flat for years and I want my money back (expected RERA:18; hybrid_all_gated rank: None)
- q44: Where can I complain against a property developer who has not delivered my apartment (expected RERA:31; hybrid_all_gated rank: None)
- q45: I want to end my marriage and get a divorce (expected HMA:13; hybrid_all_gated rank: None)
- q46: Can a daughter claim a share in her father's ancestral family property (expected HSA:6; hybrid_all_gated rank: 2)
- q47: A man sexually abused a ten year old child (expected POCSO:4, POCSO:7; hybrid_all_gated rank: None)
- q48: I cannot afford a lawyer, can I get free legal help (expected LSA:12; hybrid_all_gated rank: None)
