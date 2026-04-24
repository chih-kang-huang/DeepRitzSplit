# from utils.pf.wavelets import wavedec2, waverec2
# 
# 
# class WaveConv2d(eqx.Module):
#     in_channels: int
#     out_channels: int
#     weights: List[jnp.Array]
#     wavelet: str
#     mode: str
#     level: int | None
# 
#     def __init__(self, in_channels, out_channels,  N_x, N_y, 
#                 level = 1,
#                 wavelet = 'db12', mode='periodization', *, key
#                 ):
# 
#         """
#         2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
#         
#         Input parameters: 
#         -----------------
#         in_channels  : scalar, input kernel dimension
#         out_channels : scalar, output kernel dimension
#         level        : scalar, levels of wavelet decomposition
#         size         : scalar, length of input 1D signal
#         wavelet      : string, wavelet filters
#         mode         : string, padding style for wavelet decomposition
#         
#         It initializes the kernel parameters: 
#         -------------------------------------
#         self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
#                         kernel weights for Approximate wavelet coefficients
#         self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
#                         kernel weights for Horizontal-Detailed wavelet coefficients
#         self.weights3 : tensor, shape-[in_channels * out_channels * x * y]
#                         kernel weights for Vertical-Detailed wavelet coefficients
#         self.weights4 : tensor, shape-[in_channels * out_channels * x * y]
#                         kernel weights for Diagonal-Detailed wavelet coefficients
#         """
# 
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.level = level
#         # if isinstance(size, list):
#         #     if len(size) != 2:
#         #         raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
#         #     else:
#         #         self.size = size
#         # else:
#         #     raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
# 
#         self.wavelet = wavelet       
#         self.mode = mode
#         key, dummykey = jr.split(key)
#         dummy_data = jr.uniform(dummykey, shape=(N_x, N_y))        
#         A, *HVD = wavedec2(dummy_data, level=self.level, mode=self.mode, wavelet=self.wavelet)
#         self.modes1 = A.shape[-2]
#         self.modes2 = A.shape[-1]
#         
#         # Parameter initilization
#         scale = (1 / (in_channels * out_channels))
#         self.weights = []
#         for _ in range(4): 
#             key, wkey = jr.split(key)
#             self.weights.append( 
#                 scale * jr.normal(wkey, 
#                                   shape=(in_channels, out_channels, self.modes1, self.modes2))
#             )
# 
#     # Convolution
#     def mul2d(self, 
#               input : Float[Array, "in_channel x y"], 
#               weights :  Float[Array, "in_channel out_channel x y"], 
#               ):
#         """
#         Performs element-wise multiplication
# 
#         Input Parameters
#         ----------------
#         input   : tensor, shape-(batch * in_channel * x * y )
#                   2D wavelet coefficients of input signal
#         weights : tensor, shape-(in_channel * out_channel * x * y)
#                   kernel weights of corresponding wavelet coefficients
# 
#         Returns
#         -------
#         convolved signal : tensor, shape-(out_channel * x * y)
#         """
#         return jnp.einsum("ixy,ioxy->oxy", input, weights)
# 
#     @eqx.filter_jit
#     def __call__(self, 
#                 x : Float[Array, "Channel x y"]):
#         """
#         Input parameters: 
#         -----------------
#         x : tensor, shape-[Channel * x * y]
#         Output parameters: 
#         ------------------
#         x : tensor, shape-[Channel * x * y]
#         """
#         if x.shape[-1] > self.size[-1]:
#             factor = int(np.log2(x.shape[-1] // self.size[-1]))
#             
#             # Compute single tree Discrete Wavelet coefficients using some wavelet
#             #dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet)
#             x_ft, *x_coeff = wavedec2(x, wavelet=self.wavelet, level = self.level + factor, mode=self.mode)
#             
#         elif x.shape[-1] < self.size[-1]:
#             factor = int(np.log2(self.size[-1] // x.shape[-1]))
#             
#             # Compute single tree Discrete Wavelet coefficients using some wavelet
#             x_ft, *x_coeff = wavedec2(x, wavelet=self.wavelet, level = self.level - factor, mode=self.mode)
#         
#         else:
#             # Compute single tree Discrete Wavelet coefficients using some wavelet
#             x_ft, *x_coeff = wavedec2(x, wavelet=self.wavelet, level = self.level, mode=self.mode)
# 
#         # Instantiate higher level coefficients as zeros
#         out_coeff = [jnp.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
#         
#         # Multiply the final approximate Wavelet modes
#         A_out = self.mul2d(x_ft, self.weights[0])
#         # Multiply the final detailed wavelet coefficients
#         out_coeff[-1][:,0,:,:] = self.mul2d(x_coeff[-1][:,0,:,:].clone(), self.weights[1])
#         out_coeff[-1][:,1,:,:] = self.mul2d(x_coeff[-1][:,1,:,:].clone(), self.weights[2])
#         out_coeff[-1][:,2,:,:] = self.mul2d(x_coeff[-1][:,2,:,:].clone(), self.weights[3])
#         
#         # Return to physical space        
#         # idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
#         # x = idwt((out_ft, out_coeff))
#         # return x
# 
