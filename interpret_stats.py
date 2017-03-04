import pstats

p = pstats.Stats('teststats')
#p.sort_stats('tottime').print_stats()
p.print_callees('train').sort_stats('tottime')
