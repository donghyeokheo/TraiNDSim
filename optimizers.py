import torch
from conductance_formula import *

def pulse_reset(args, p, grad, delta_p):
    p_reset = p.clone()
    flag = torch.logical_or(p[0] > args.MaxPulse, p[1] > args.MaxPulse)
    grad = grad[flag].clone()

    # increased weights
    p_reset[0][flag][grad <= 0] = p[0][flag][grad <= 0] - (args.MaxPulse - p[1][flag][grad <= 0])
    p_reset[1][flag][grad <= 0] = args.MaxPulse

    # decreased weights
    p_reset[1][flag][grad > 0] = p[1][flag][grad > 0] - (args.MaxPulse - p[0][flag][grad > 0])
    p_reset[0][flag][grad > 0] = args.MaxPulse
    
    p_reset[p_reset > args.MaxPulse] = args.MaxPulse

    # modifying delta pulse
    delta_p[0][flag][grad <= 0] = p_reset[0][flag][grad <= 0].clone()
    delta_p[1][flag][grad <= 0] = args.MaxPulse - p[1][flag][grad <= 0]
    delta_p[1][flag][grad > 0] = p_reset[1][flag][grad > 0].clone()
    delta_p[0][flag][grad > 0] = args.MaxPulse - p[0][flag][grad > 0]

    return p_reset, delta_p

def conductance_updating(args, grad, cdt, nl_LTP, nl_LTD, ns):
    p = torch.zeros_like(cdt)
    updated_p = torch.zeros_like(cdt)
    updated_cdt = torch.zeros_like(cdt)

    p[0][grad <= 0] = pulse_LTP(args, cdt[0][grad <= 0], nl_LTP[0][grad <= 0])
    p[0][grad > 0] = pulse_LTD(args, cdt[0][grad > 0], nl_LTD[0][grad > 0])
    if args.weight_representation == 'bi':
        p[1][grad <= 0] = pulse_LTD(args, cdt[1][grad <= 0], nl_LTD[1][grad <= 0])
        p[1][grad > 0] = pulse_LTP(args, cdt[1][grad > 0], nl_LTP[1][grad > 0])

    delta_p = args.learning_rate * grad.abs() * args.MaxPulse / (2*ns)
    if args.delta_pulse_representation == 1:
        max_grad = torch.max(torch.abs(grad))
        if max_grad == 0:
            # print("Max gradient is zero.")
            max_grad = max_grad + 1e-8
        delta_p = delta_p / max_grad
    
    if args.pulse_update == 'one':
        # add probability
        noise = torch.cuda.FloatTensor(*grad.size()).uniform_()
        delta_p = torch.floor(delta_p + noise)
        delta_p[delta_p > 1] = 1

        # # add threshold
        # threshold = 0.5
        # delta_p[delta_p > threshold] = 1

    delta_p = delta_p.floor()

    updated_p[0] = p[0] + delta_p
    if args.weight_representation == 'bi':
        updated_p[1] = p[1] + delta_p
        # pulse reset
        if delta_p.dim() == 2:
            delta_p = torch.tile(delta_p, (2,1,1))
        if delta_p.dim() == 4:
            delta_p = torch.tile(delta_p, (2,1,1,1,1))
        updated_p, delta_p = pulse_reset(args, updated_p, grad, delta_p)
    else:
        updated_p[updated_p > args.MaxPulse] = args.MaxPulse
        delta_p = updated_p[0] - p[0]

    # apply cycle-to-cycle variation (sigma_normalization / G ~ [0,1])
    c2c = torch.normal(torch.zeros_like(cdt), args.c2c_vari*torch.ones_like(cdt))
    c2c_flag = torch.sign(torch.abs(grad))
    
    updated_cdt[0][grad <= 0] = conductance_LTP(args, updated_p[0][grad <= 0], nl_LTP[0][grad <= 0])
    updated_cdt[0][grad > 0] = conductance_LTD(args, updated_p[0][grad > 0], nl_LTD[0][grad > 0])
    if args.weight_representation == 'bi':
        updated_cdt[0] += torch.sqrt(delta_p[0]) * c2c[0] * c2c_flag * (args.Gmax-args.Gmin) / ns
        updated_cdt[1][grad <= 0] = conductance_LTD(args, updated_p[1][grad <= 0], nl_LTD[1][grad <= 0])
        updated_cdt[1][grad > 0] = conductance_LTP(args, updated_p[1][grad > 0], nl_LTP[1][grad > 0])
        updated_cdt[1] += torch.sqrt(delta_p[1]) * c2c[1] * c2c_flag * (args.Gmax-args.Gmin) / ns
        updated_w = (updated_cdt[0] - updated_cdt[1]) / (args.Gmax-args.Gmin) * ns
    else:
        updated_cdt[0] += torch.sqrt(delta_p) * c2c[0] * c2c_flag * ((args.Gmax-args.Gmin)/2) / ns
        updated_cdt[1] = cdt[1]
        updated_w = (updated_cdt[0] - updated_cdt[1]) / ((args.Gmax-args.Gmin)/2) * ns

    return updated_w, updated_cdt

def SGD(args, weight, gradient, conductance, nl_LTP, nl_LTD, normalization_scale):
    gradient = gradient + args.weight_decay*weight/args.batch_size
    updated_weight, updated_conductances = conductance_updating(args, gradient, conductance, nl_LTP, nl_LTD, normalization_scale)

    return updated_weight, updated_conductances

def Momentum(args, weight, gradient, conductance, nl_LTP, nl_LTD, normalization_scale, m, step):
    gradient = gradient + args.weight_decay*weight/args.batch_size
    if step > 1:
        m = args.beta*m + (1-args.dampening)*gradient
    else:
        m = gradient

    updated_weight, updated_conductances = conductance_updating(args, m, conductance, nl_LTP, nl_LTD, normalization_scale)
    m = weight - updated_weight
    # m = (weight - updated_weight) / args.learning_rate

    return updated_weight, updated_conductances, m

def Adam(args, weight, gradient, conductance, nl_LTP, nl_LTD, normalization_scale, m, v, step):
    gradient = gradient + args.weight_decay*weight/args.batch_size

    m = args.betas[0]*m + (1-args.betas[0])*gradient
    v = args.betas[1]*v + (1-args.betas[1])*(gradient*gradient)
    m_correction = m / (1-args.betas[0]**step)
    v_correction = v / (1-args.betas[1]**step)

    gradient = m_correction / (torch.sqrt(v_correction) + args.eps)

    updated_weight, updated_conductances = conductance_updating(args, gradient, conductance, nl_LTP, nl_LTD, normalization_scale)
    # gradient = weight - updated_weight
    # gradient = (weight - updated_weight) / args.learning_rate
    # m = gradient * (torch.sqrt(v_correction) + args.eps) * (1-args.betas[0]**step)

    return updated_weight, updated_conductances, m, v

