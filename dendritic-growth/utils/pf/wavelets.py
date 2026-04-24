import equinox as eqx
import cr.wavelets as wt

@eqx.filter_jit
def wavedec2(phi, wavelet, level, mode):
    AHVD=[]
    A = phi
    for _ in range(level): 
        A, HVD_coeff = wt.dwt2(A, wavelet=wavelet, mode=mode)
        AHVD.insert(0, HVD_coeff)
    AHVD.insert(0, A) 
    return AHVD

@eqx.filter_jit
def waverec2(AHVD, wavelet, level, mode):
    A, *HVD = AHVD
    for _ in range(level): 
        A = wt.idwt2([A] + HVD[0:1], wavelet = wavelet, mode =mode)
        HVD = HVD[1:]
    return A