from imports import *


def freeze(m):
    """
    Freezes a model
    
    Arguments:
        m {torch module} -- Model to freeze
    """
    for param in m.parameters():
        param.requires_grad = False
    unfreeze_bn(m)


def unfreeze(m):
    """
    Unreezes a model
    
    Arguments:
        m {torch module} -- Model to unfreeze
    """
    for param in m.parameters():
        param.requires_grad = True


def freeze_layer(model, name):
    """
    Freezes layer(s) of a model
    
    Arguments:
        model {[torch module]} -- Model to freeze layers from
        name {[str]} -- Layers containing "name" in their name will be frozen
    """
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = False


def unfreeze_layer(model, name):
    """
    Unfreezes layer(s) of a model
    
    Arguments:
        model {[torch module]} -- Model to unfreeze layers from
        name {[str]} -- Layers containing "name" in their name will be unfrozen
    """
    for n, p in list(model.named_parameters()):
        if name in n:
            p.requires_grad = True