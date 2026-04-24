from utils.models.FNO import FNO2d
from utils.models.UNet import UNet, ResUNet
from utils.models.DR import DR1Block
from utils.models.hybrid_models import RDNOPrescribed, RDNOPL
import jax



def NNselection2D(heading, params, activation, key, comment = None, **kwargs):
    Archtecture = params['Architecture']
    model_path = f"{heading}-{Archtecture}"
    if comment :
        model_path += f"-{comment}"

    match Archtecture:
        case "FNO":
            in_channels = params['in_channels']
            out_channels = params['out_channels']
            modes1 = params['modes1']
            modes2 = params['modes2']
            width = params['width']
            n_layers = params['n_layers']
            NN= FNO2d(
                in_channels=in_channels, out_channels=out_channels, modes1=modes1, modes2=modes2, width=width, activation=activation, n_blocks = n_layers, key=key
            )
            model_path += f"-m{modes1}-w{width}-lv{n_layers}"
        case "DR1":
            in_channels = params['in_channels']
            out_channels = params['out_channels']
            kernel_size = params['kernel_size']
            width = params['width']
            n_layers = params['n_layers']
            NN= DR1Block(
                in_channels=in_channels, out_channels=out_channels, width=width, kernel_size=kernel_size, n_layers = n_layers,
                activation=activation, key = key,
            )
            model_path += f"-kr{kernel_size}-w{width}-lv{n_layers}"
        case "UNet":
            in_channels = params['in_channels']
            out_channels = params['out_channels']
            hidden_channels = params['hidden_channels']
            #width = params['width']
            n_levels = params['n_levels']
            NN= UNet(
                num_spatial_dims=2, in_channels=in_channels, out_channels=out_channels, 
                hidden_channels= hidden_channels, num_levels= n_levels, 
                activation = activation,
                key = key
            )
            #model_path += f"-h{hidden_channels}-lv{n_levels}-w{width}"
            model_path += f"-h{hidden_channels}-lv{n_levels}"
        case "ResUNet":
            hidden_channels = params['hidden_channels']
            n_levels = params['n_levels']
            NN= ResUNet(
                2, 2, 1, hidden_channels= hidden_channels, num_levels= n_levels, 
                activation = activation,
                key = key
                )
            model_path += f"-h{hidden_channels}-lv{n_levels}"
        
        case "RDNOP":
            params_diffuse = params['params_diffuse']
            archi_diffuse = params_diffuse['Architecture']
            key, dkey =jax.random.split(key)
            diffuseBlock, model_path = NNselection2D(
                heading=heading, params=params_diffuse, activation=activation, key = dkey
            )
            model_path = model_path.replace(archi_diffuse, Archtecture)

            nonlinear = kwargs['nonlinear']
            NN = RDNOPrescribed(
                diffuseBlock= diffuseBlock, nonlinear= nonlinear,
                key=key,
            )
        case "RDNOPL":
            params_diffuse = params['params_diffuse']
            archi_diffuse = params_diffuse['Architecture']
            width = params_diffuse['in_channels']
            key, dkey =jax.random.split(key)
            diffuseBlock, model_path = NNselection2D(
                heading=heading, params=params_diffuse, activation=activation, key = dkey
            )
            model_path = model_path.replace(archi_diffuse, Archtecture)
            model_path += f"-w{width}"

            nonlinear = kwargs['nonlinear']
            NN = RDNOPL(
                width=width,
                diffuseBlock= diffuseBlock, nonlinear= nonlinear,
                key=key,
            )

        #case "ReactDiff": 
        #    from utils.models.NNBlocks import ReactionBlock
        #    #from utils.models.UNet import DoubleConv2d
        #    from utils.models.hybrid_models import ReactionDiffusion2D, ReactionDiffusionFFT
        #    from utils.models.MLP import MLP2D
        #    key, rkey = jr.split(key)


        #    width_react = 2 
        #    n_layers_react = 2 
        #    ReactNN = ReactionBlock(
        #        in_channels = 2, out_channels = 1, width= width_react, n_layers = n_layers_react, 
        #        activation =jax.nn.tanh, key = rkey
        #    )
        #    # n_hidden = 2
        #    # n_neurons = 64
        #    # ReactNN = MLP2D(
        #    #     2, 'scalar', n_hidden, n_neurons, key=rkey
        #    # )
        #    key, dkey = jr.split(key)
        #    diffuse = "UNet"
        #        case "UNet" :
        #            from utils.models.UNet import UNet
        #            DiffuseNN = UNet(
        #                2, 1, 1, hidden_channels= hidden_channels, num_levels= num_levels, 
        #                activation = (jax.nn.tanh), 
        #                key = key
        #            )
        #            model_path += f"-{diffuse}-h{hidden_channels}-lv{num_levels}"
        #        case "2Conv": 
        #            from utils.models.NNBlocks import ReactionBlock
        #            from utils.models.UNet import DoubleConv2d
        #            from utils.models.hybrid_models import ReactionDiffusion2D
        #            from utils.models.MLP import MLP2D
        #            class DiffuseConv2D(eqx.Module):
        #                DiffuseBlocks : list[DoubleConv2d]

        #                def __init__(
        #                        self, in_channels, out_channels, n_levels, kernel_size, padding, stride, activation,
        #                        *, key
        #                ):
        #                    self.DiffuseBlocks = []
        #                    for _ in range(n_levels) : 
        #                        key, dkey =jr.split(key)
        #                        self.DiffuseBlocks.append(
        #                            DoubleConv2d(
        #                                in_channels, out_channels, kernel_size, padding=padding, stride = stride, activation=activation, key =dkey
        #                            )
        #                        )
        #                @eqx.filter_jit
        #                def __call__(
        #                        self, x
        #                ): 
        #                    for Diffuse in self.DiffuseBlocks:
        #                        x = Diffuse(x)

        #                    #return x.squeeze(0) if x.shape[0] == 1 else x
        #                    return x

        #            width = 2 
        #            n_layers = 2 
        #            kernel_size = 31  #17
        #            padding = kernel_size //2
        #            stride = 2 
        #            n_levels = 2
        #            DiffuseNN = DiffuseConv2D(
        #                in_channels=1, out_channels=1, n_levels=n_levels, kernel_size=kernel_size, padding=padding, stride =stride, activation=jax.nn.tanh, key = dkey
        #            )
        #            model_path += f"-{diffuse}-kr{kernel_size}-l{n_levels}-p{padding}-s{stride}"
        #    # NN= ReactionDiffusion2D(
        #    #     ReactNN, DiffuseNN
        #    # )

        #    NN= ReactionDiffusionFFT(
        #        ReactNN, DiffuseNN
        #    )
        #    model_path = model_path.replace(f"{diffuse}", f"{diffuse}-fft")

        case _:
            raise Exception(f"Architecture {Archtecture} not implemented yet !")


    return NN, model_path
