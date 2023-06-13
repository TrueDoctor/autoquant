single bit:

min (e1[1], e2[1])

two bits:
min e1[2], e2[2], e1[1] + e2[1]

= min e1[1] + min(e1[+1], e2[+2]), e2[1] + min(e1[+1], e2[+1])

three bits:
min e1[3], e2[3], e1[2] + e2[1], e1[1] + e2[2]

= min e1[2] + min(e1[+1], e2[+1]), e2[2] + min(e1[+1], e2[+1])

four bits: 

min e1[4], e2[4], e1[3] + e2[1], e1[1] + e2[3], e1[2] + e2[2]

= min e1[3] + min(e1[+1], e2[+1]), e2[3] + min(e1[+1], e2[+1])
