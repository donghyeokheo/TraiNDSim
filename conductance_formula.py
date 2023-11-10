import numpy as np
import torch

def conductance_LTP(args, pulse, nl_LTP):
    if args.FormulaType == 0:
        return args.Gmin + (args.Gmax-args.Gmin)/nl_LTP * torch.log((torch.exp(nl_LTP)-1)/args.MaxPulse * pulse + 1)

    elif args.FormulaType == 1:
        return args.Gmin + (args.Gmax-args.Gmin)/(1 - torch.exp(-nl_LTP)) * (1 - torch.exp(-nl_LTP/args.MaxPulse * pulse))
    
    elif args.FormulaType == 2:
        return args.Gmin + ((args.Gmax-args.Gmin)/(torch.exp(nl_LTP) - 1)) * ((torch.exp(nl_LTP)+1)/(1+torch.exp(-nl_LTP*(2/args.MaxPulse * pulse - 1))) - 1)

def conductance_LTD(args, pulse, nl_LTD):
    if args.FormulaType == 0:
        return args.Gmax - (args.Gmax-args.Gmin)/nl_LTD * torch.log((torch.exp(nl_LTD)-1)/args.MaxPulse * pulse + 1)  
    
    elif args.FormulaType == 1:
        return args.Gmax - (args.Gmax-args.Gmin)/(1 - torch.exp(-nl_LTD)) * (1 - torch.exp(-nl_LTD/args.MaxPulse * pulse))

    elif args.FormulaType == 2:
        return args.Gmax - ((args.Gmax-args.Gmin)/(torch.exp(nl_LTD) - 1)) * ((torch.exp(nl_LTD)+1)/(1+torch.exp(-nl_LTD*(2/args.MaxPulse * pulse - 1))) - 1)
    
def pulse_LTP(args, conductance, nl_LTP):
    if args.FormulaType == 0:
        return args.MaxPulse/(torch.exp(nl_LTP)-1) * (torch.exp(nl_LTP*(conductance-args.Gmin)/(args.Gmax-args.Gmin)) - 1)
    
    elif args.FormulaType == 1:
        return -args.MaxPulse/nl_LTP * torch.log(1 - (conductance-args.Gmin)/(args.Gmax-args.Gmin)*(1-torch.exp(-nl_LTP)))
    
    elif args.FormulaType == 2:
        return args.MaxPulse/2 * (1 - 1/nl_LTP * torch.log((torch.exp(nl_LTP)+1)/((conductance-args.Gmin)/(args.Gmax-args.Gmin) * (torch.exp(nl_LTP)-1) + 1) - 1))

def pulse_LTD(args, conductance, nl_LTD):
    if args.FormulaType == 0:
        return args.MaxPulse/(torch.exp(nl_LTD)-1) * (torch.exp(nl_LTD*(args.Gmax-conductance)/(args.Gmax-args.Gmin)) - 1)
    
    elif args.FormulaType == 1:
        return -args.MaxPulse/nl_LTD * torch.log(1 - (args.Gmax-conductance)/(args.Gmax-args.Gmin)*(1-torch.exp(-nl_LTD)))

    elif args.FormulaType == 2:
        return args.MaxPulse/2 * (1 - 1/nl_LTD * torch.log((torch.exp(nl_LTD)+1)/((args.Gmax-conductance)/(args.Gmax-args.Gmin) * (torch.exp(nl_LTD)-1) + 1) - 1))



