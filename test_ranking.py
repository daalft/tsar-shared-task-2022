from ranking.ls_ranker import DefaultEnglishLSRanker
# from ranking.ls_ranker import DefaultSpanishLSRanker

ltr = ["concealed", "dressed", "hidden", "camouflaged", "changed", "covered", "disguised", "masked", "unrecognizable", "converted", "impersonated"]
ranker_en = DefaultEnglishLSRanker()
ranker_en.rank(ltr)

# ranker_es = DefaultSpanishLSRanker()
# ranker_es.rank(ltr_es)