# This software is distributed under the 3-clause BSD License.
'''
markshare2 instance from MIPLIB 2017:
    https://miplib.zib.de/instance_details_markshare2.html
'''
import pyomo.environ as pyo

m = pyo.ConcreteModel()

m.xc = pyo.Var(list(range(1,15)), within=pyo.NonNegativeReals)
m.xb = pyo.Var(list(range(15,75)), within=pyo.Binary)

var_dict = {}
for idx in m.xc:
    var_dict[f'x{idx}'] = m.xc[idx]
for idx in m.xb:
    var_dict[f'x{idx}'] = m.xb[idx]

locals().update(var_dict)

m.obj = pyo.Objective(expr=x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 + x11 - x12 + x13 - x14)

m.c = pyo.Constraint(list(range(1,8)))
m.c[1] = ( x1 + x2 + 74*x15 + 49*x16 + 12*x17 + 93*x18 + 56*x19 + 16*x20 + 39*x21
     + 77*x22 + 56*x23 + 73*x24 + x25 + 3*x26 + 68*x27 + 61*x28 + 8*x29
     + 55*x30 + 18*x31 + 21*x32 + 57*x33 + 98*x34 + 58*x35 + 57*x36 + 46*x37
     + 72*x38 + 6*x39 + 16*x40 + 76*x41 + 21*x42 + 78*x43 + 18*x44 + 11*x45
     + 58*x46 + 59*x47 + 25*x48 + 32*x49 + 14*x50 + 16*x51 + 3*x52 + 60*x53
     + 12*x54 + 7*x55 + 42*x56 + 98*x57 + 34*x58 + 33*x59 + 16*x60 + 97*x61
     + 63*x62 + 66*x63 + 28*x64 + 57*x65 + 19*x66 + 74*x67 + 44*x68 + 45*x69
     + 49*x70 + 76*x71 + 74*x72 + 9*x73 + 44*x74  == 1324 )
m.c[2] = ( x3 + x4 + 20*x15 + 7*x16 + 68*x17 + 69*x18 + 95*x19 + 64*x20 + 76*x21
     + 12*x22 + 45*x23 + 43*x24 + 83*x25 + 15*x26 + 90*x27 + 10*x28 + 96*x29
     + 98*x30 + 53*x31 + x32 + 2*x33 + 58*x34 + 24*x35 + 90*x36 + 29*x37
     + 57*x38 + 19*x39 + 73*x40 + 89*x41 + 31*x42 + 12*x43 + 34*x44 + 67*x45
     + 48*x46 + 11*x47 + 22*x48 + 36*x49 + 78*x50 + 75*x51 + 52*x52 + 95*x53
     + 57*x54 + 62*x55 + 94*x56 + 10*x57 + 42*x58 + 89*x59 + 11*x60 + 77*x61
     + 85*x62 + 30*x63 + 82*x64 + 20*x65 + 52*x66 + 78*x67 + 6*x68 + 57*x69
     + 65*x70 + 79*x71 + 83*x72 + 16*x73 + 67*x74 == 1554 )
m.c[3] = ( x5 + x6 + 85*x15 + 47*x16 + 67*x17 + 59*x18 + 84*x19 + 59*x20 + 19*x21
     + 8*x22 + 50*x23 + 66*x24 + 5*x25 + 51*x26 + 51*x27 + 64*x28 + 64*x29
     + 53*x30 + 61*x31 + 45*x32 + 3*x33 + 76*x34 + 17*x35 + 54*x36 + 13*x37
     + 89*x38 + 68*x39 + 57*x40 + 4*x41 + 24*x42 + 96*x43 + 81*x44 + 36*x45
     + 54*x46 + 3*x47 + 82*x48 + 33*x49 + 88*x50 + x51 + 29*x52 + 4*x53
     + 48*x54 + 51*x55 + 14*x56 + 86*x57 + 64*x58 + 73*x59 + 78*x60 + 45*x61
     + 65*x62 + 30*x63 + 52*x64 + 6*x65 + 78*x66 + 9*x67 + 19*x68 + 87*x69
     + 73*x70 + 10*x71 + 87*x72 + 33*x73 + x74 == 1429 )
m.c[4] = ( x7 + x8 + 13*x15 + 71*x16 + 78*x17 + 84*x18 + 56*x19 + 66*x20 + 8*x21
     + 68*x22 + 48*x23 + 28*x24 + 33*x25 + 34*x26 + 8*x27 + 99*x28 + 80*x29
     + 74*x30 + 2*x31 + 10*x32 + 96*x33 + 41*x34 + 98*x35 + 74*x36 + 39*x37
     + 91*x38 + 85*x39 + 95*x40 + 96*x41 + x42 + 80*x43 + 90*x44 + 97*x45
     + 36*x46 + 7*x47 + 69*x48 + 9*x49 + 9*x50 + 93*x51 + 94*x52 + 44*x53
     + 36*x54 + 71*x55 + 37*x56 + 72*x57 + 38*x58 + 74*x59 + 89*x60 + 37*x61
     + 24*x62 + 88*x63 + 77*x64 + 61*x65 + 80*x66 + 2*x67 + 60*x68 + 87*x69
     + 80*x70 + 74*x71 + 42*x72 + 2*x73 + 37*x74  == 1686 )
m.c[5] = ( x9 + x10 + 35*x15 + 61*x16 + 66*x17 + 78*x18 + 46*x19 + 89*x20 + 61*x21
     + 25*x22 + 55*x23 + 16*x24 + 81*x25 + 35*x26 + 96*x27 + 23*x28 + 83*x29
     + 39*x30 + 14*x31 + 53*x32 + 23*x33 + 23*x34 + 93*x35 + 38*x36 + 15*x37
     + 20*x38 + 19*x39 + 28*x40 + 79*x41 + 51*x42 + 24*x43 + 6*x44 + 3*x45
     + 47*x46 + 61*x47 + 60*x48 + 71*x49 + 63*x50 + 26*x51 + 66*x52 + 71*x53
     + 63*x54 + 56*x55 + 32*x56 + 39*x57 + 31*x58 + 64*x59 + 89*x60 + 62*x61
     + 68*x62 + 59*x63 + 71*x64 + 48*x65 + 76*x66 + 96*x67 + 74*x68 + 61*x69
     + 21*x70 + 46*x71 + 18*x72 + 23*x73 + 24*x74 == 1482 )
m.c[6] = ( x11 + x12 + 86*x15 + 8*x16 + 44*x17 + 96*x18 + 64*x19 + 65*x20 + 68*x21
     + 53*x22 + 19*x23 + 33*x24 + 28*x25 + 42*x26 + 72*x27 + 39*x28 + 5*x29
     + 77*x30 + 37*x31 + 89*x32 + 7*x33 + 78*x34 + 10*x35 + 78*x36 + 10*x37
     + 96*x38 + 55*x39 + x40 + 64*x41 + 61*x42 + 63*x43 + 90*x44 + 22*x45
     + 78*x46 + 92*x47 + 25*x48 + 24*x49 + 65*x50 + 6*x51 + 68*x52 + 66*x53
     + 66*x54 + x55 + 67*x56 + 78*x57 + 21*x58 + 47*x59 + 17*x60 + 89*x61
     + 77*x62 + 88*x63 + 54*x64 + 10*x65 + 87*x66 + 88*x67 + 80*x68 + 76*x69
     + 9*x70 + 83*x71 + 95*x72 + 86*x73 + 24*x74 == 1613 )
m.c[7] = ( x13 + x14 + 41*x15 + 64*x16 + 82*x17 + 24*x18 + 48*x19 + 41*x20 + 29*x21
     + 93*x22 + 64*x23 + 39*x24 + 92*x25 + 86*x26 + 64*x27 + 45*x28 + 87*x29
     + 34*x30 + 39*x31 + 88*x32 + 99*x33 + 63*x34 + 85*x35 + 48*x36 + 83*x37
     + 88*x38 + 85*x39 + 5*x40 + 14*x41 + 31*x42 + 12*x43 + 93*x44 + 55*x45
     + x46 + 2*x47 + 22*x48 + 93*x49 + 49*x50 + 35*x51 + 25*x52 + 39*x53 + x54
     + 77*x55 + 43*x56 + 7*x57 + 42*x58 + 36*x59 + 63*x60 + 5*x61 + 8*x62
     + 43*x63 + 18*x64 + 60*x65 + 47*x66 + 47*x67 + 46*x68 + 45*x69 + 38*x70
     + 9*x71 + 37*x72 + 8*x73 + 82*x74 == 1424 )

x2.value = 0
x2.fix()
x4.value = 0
x4.fix()
x6.value = 0
x6.fix()
x8.value = 0
x8.fix()
x10.value = 0
x10.fix()
x12.value = 0
x12.fix()
x14.value = 0
x14.fix()

model = m
