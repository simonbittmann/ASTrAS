# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:29:15 2020

@author: bittmans
"""

import re
import numpy as np

import random

from scipy.special import erf, wofz
from scipy.signal import find_peaks
from .helpers import BlankObject, split_string_into_lines
import lmfit


# %%
# general fit methods and classes
# basic fit functions


def gauss_conv_exp(t, p, sigma, *args, t0=0, **kwargs):
    # exponential decay in t (time) with decay parameter p,
    # convoluted with a gaussian with parameter sigma, centered at t0
    return (0.5*np.exp(-(t-t0)/p)*np.exp((sigma**2)/(2*(p**2))) *
            (1 + erf(((t-t0) - (sigma**2)/p)/(np.sqrt(2)*sigma))))


def exp(t, p, *args, t0=0, **kwargs):
    return np.exp(-(t-t0)/p)


def sine_modul_exp(self, t, p, damp=True, sigma=None,
                   t0=0):
    # sine modulated exponential decay (convoluted if bool(sigma) is True)
    ex = gauss_conv_exp if sigma else exp
    f = ex((t-t0), p[0], sigma)
    if damp:
        f = f + p[1]*np.sin(p[2]*(t-t0) - p[3])*ex((t-t0), p[4], sigma)
    else:
        f = f + p[1]*np.sin(p[2]*(t-t0) - p[3])
    return f


def gaussian(x, x0, sig):
    return np.exp(-0.5*((x-x0)**2)/sig**2)


def lorentzian(x, x0, gamma):
    return 1/(np.pi*gamma)*(gamma**2/((x - x0)**2 + gamma**2))


# model function class made for Global fit,
# methods can also be used for line fit, picking specific components
class FitModels():
    def __init__(self, max_parallel_comp=8):
        # branched kinetic models
        # 1)    A->B + A->C
        # 2)    A->B; B->C + B->D
        # 3)    A->B + A->C; B->D + C->D
        # 4)    A->B; B->C + B->D; D->E + C->E
        # see following dict for the mechanisms of the remaining models
        ####
        #   Components are sorted in alphabetical order before returned by
        #   the respective fit function. Classes using this class will likely
        #   use tau instead of k, with parameters tau being numbered tau_1,
        #   tau_2, etc. for a translation into k_AB, k_BC etc. see following
        #   dict (tau_order) or notes in specific fit function
        #
        self.model_dict = {
            'AB|AC': {'mechanism': "A->B + A->C",
                      'tau_order': ["AB", "AC", "B", "C"],
                      'number_of_species': 3,
                      'number_of_decays': 4,
                      'func': self._kinetic_branch_one},
            'ABC|ABD': {'mechanism': "A->B; B->C + B->D",
                        'tau_order':  ["AB", "BC", "BD", "C", "D"],
                        'number_of_species': 4,
                        'number_of_decays': 5,
                        'func': self._kinetic_branch_two},
            'ABD|ACD': {'mechanism': "A->B + A->C; B->D + C->D",
                        'tau_order': ["AB", "AC", "BD", "CD", "D"],
                        'number_of_species': 4,
                        'number_of_decays': 5,
                        'func': self._kinetic_branch_three},
            'ABCE|BDE': {'mechanism': "A->B; B->C + B->D; D->E + C->E",
                         'tau_order': ["AB", "BC", "BD", "CE", "DE", "E"],
                         'number_of_species': 5,
                         'number_of_decays': 6,
                         'func': self._kinetic_branch_four},
            'ABC|AC': {'mechanism': "A->B + A->C; B->C",
                       'tau_order': ["AB", "AC", "BC", "C"],
                       'number_of_species': 3,
                       'number_of_decays': 4,
                       'func': self._kinetic_branch_five},
            'ABCD|ABCE': {'mechanism': "A->B; B->C; C->D + C->E",
                          'tau_order': ["AB", "BC", "CD", "CE", "D", "E"],
                          'number_of_species': 5,
                          'number_of_decays': 6,
                          'func': self._kinetic_branch_six},
            'ABC|D': {'mechanism': "A->B; B->C; D->None",
                      'tau_order': ["AB", "BC", "C", "D"],
                      'number_of_species': 4,
                      'number_of_decays': 4,
                      'func': self._kinetic_branch_seven},
            'ACDE|BCDE': {'mechanism': "A->C, B->C; C->D; D->E",
                          'tau_order': ["AC", "BC", "CD", "DE", "E"],
                          'number_of_species': 5,
                          'number_of_decays': 5,
                          'func': self._kinetic_branch_eight},
            'ABCD|E': {'mechanism': "A->B; B->C; C->D; E->None",
                      'tau_order': ["AB", "BC", "CD", "D", "E"],
                      'number_of_species': 5,
                      'number_of_decays': 5,
                      'func': self._kinetic_branch_nine},
            'ABCDE|F': {'mechanism': "A->B; B->C; C->D; D->E; F->None",
                      'tau_order': ["AB", "BC", "CD", "DE", "E", "F"],
                      'number_of_species': 6,
                      'number_of_decays': 6,
                      'func': self._kinetic_branch_ten},
            
            'ABC|AD': {'mechanism': "A->B + A->D; B->C",
                      'tau_order': ["AB", "AD", "BC", "C", "D"],
                      'number_of_species': 4,
                      'number_of_decays': 5,
                      'func': self._kinetic_branch_eleven}}
        for key in self.model_dict.keys():
            self.model_dict[key]['category'] = "Branched Chain"
        alpha_string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # auto generate dicts for unidirectional models
        for key in ['AB', 'ABC', 'ABCD', 'ABCDE', 'ABCDEF', 'ABCDEFG', 'ABCDEFGH']:
            self.model_dict[key] = {'mechanism': "A->B; B->C; ...",
                                    'tau_order': [],
                                    'category': "Chain",
                                    'number_of_decays': len(key),
                                    'number_of_species': len(key),
                                    'func': self._kinetic_chain}
            for i in range(1, len(key)):
                self.model_dict[key]['tau_order'].append(
                    alpha_string[i-1:i+1])
            self.model_dict[key]['tau_order'].append(alpha_string[i])
        # auto generate dicts for parallel decay
        for i in range(max_parallel_comp):
            self.model_dict[" ".join([str(i + 1), "Comp."])] = {
                'mechanism': "",
                'tau_order': [str(j + 1) for j in range(i + 1)],
                'category': "Parallel",
                'number_of_decays': i + 1,
                'number_of_species': i + 1,
                'func': self._parallel}
        # adding generic parallel/sum of exponentials for compatibility
        # to be removed in future upon reworking of communication between
        # objects, moving away from a single model string to keyword args.
        self.fit_model_numbers = {}
        for i, key in enumerate(list(self.model_dict.keys())):
            self.fit_model_numbers[key] = i + 1
        self.base_func = 'exp'
        self._model = 'AB'

    def _func(self, *args, **kwargs):
        # placeholder method, will usually get overwritten at runtime level
        return

    def get_model(self):
        dct = {'model': self._model}
        try:
            dct['properties'] = self.model_dict[self._model]
        except Exception:
            pass
        return dct

    def write_model_string(self, model, case='globalfit', num_comp=2,
                           inf_comp=False, selected_comps=None):
        if selected_comps is None:
            selected_comps = []
        fit_object_str = str(num_comp)
        if inf_comp:
            fit_object_str += "inf" + fit_object_str
        fit_object_str += "model" + str(
            self.fit_model_numbers[model]) + "select"
        fit_object_str += "+".join([str(c[1]) for c in selected_comps])
        return fit_object_str

    def get_model_name(self, num):
        try:
            submodel_num = int(num)
        except Exception:
            return
        else:
            return list(self.model_dict.keys())[submodel_num - 1]

    def convert_tau_k(self, p):
        # helper function, converts life times to rate constants by inversion
        # life times need to be (lmfit) parameters named tau_1, tau_2, etc.
        # returns: 1) list of k_1, k_2, etc. 2) corresponding tau values
        k = []
        tau = []
        i = 1
        maxiter = 1000
        while i < maxiter:
            try:
                k.append(1/p['tau_' + str(i)])
            except Exception:
                break
            else:
                tau.append(p['tau_' + str(i)])
            i += 1
        return k, tau

    # set model function by name
    def set_model(self, name, base_func='exp'):
        try:
            if type(name) is int:
                name = list(self.model_dict.keys())[name - 1]
            self._set_model(self.model_dict[name]['func'],
                            base_func=base_func)
        except KeyError:
            raise
            print('Invalid model.')
        except Exception:
            self._func = self._parallel
        else:
            self._model = name

    # set model function
    def _set_model(self, f, base_func=None):
        self._func = f
        if base_func:
            if base_func in ('exp'):
                self.base_func = base_func
            else:
                print('Invalid base function for fit model, using exp.')
                self.base_func = 'exp'

    # importing branch models, only if you know what you are doing
    def import_model(self, name, func, base_func='exp'):
        if name not in self.model_dict.keys():
            self.model_dict[name]['func'] = func
            self.set_model(name, base_func=base_func)
            return True
        else:
            print(
                "\"".join(["Model ", name,
                           " already exists and cannot be overwritten."]))
            return False

    # model wrapper, containing operations universal for all models
    def fit_function(self, t, p, *args, inf_comp=False, sigma=None, **kwargs):
        kw = {}
        if self.base_func == 'exp':
            if 'num_comp' in kwargs.keys():
                kw['num_comp'] = kwargs['num_comp']
            try:
                x0 = p['x0_1'].value
            except Exception:
                try:
                    x0 = p['t0'].value
                except Exception:
                    x0 = 0
            try:
                sigma = p['sigma_1'].value
            except Exception:
                sigma = 1
                return self._func(
                    lambda *args, **kwa: exp(*args, t0=x0, **kwa), t, p, **kw)
            else:
                return self._func(
                    lambda time, para, *args, **kwa: gauss_conv_exp(
                        time, para, sigma, t0=x0, **kwa),
                    t, p, **kw)
        else:
            return

    # specific model functions
    def _parallel(self, f, t, p, num_comp=2, **kwargs):
        y = np.zeros((num_comp, len(t)))
        for i in range(num_comp):
            y[i, :] = f(t, p['tau_' + str(i + 1)])
        return y

    def _kinetic_chain(self, f, t, p, num_comp=2, inf_comp=None, **kwargs):
        def amplitudes(j, m):
            return (np.prod([para for para in k[:m]])
                    / np.prod([k[i] - k[j] for i in range(m + 1) if i != j]))
        # rate constants
        k, tau = self.convert_tau_k(p)
        # species one
        c = [np.array(f(t, p['tau_1']))]
        # following species
        for m in range(1, num_comp):
            comps = [amplitudes(j, m)*f(t, tau[j]) for j in range(m + 1)]
            c.append(np.abs(np.sum(np.array(comps), axis=0)))
        return np.array(c)

    def _kinetic_branch_one(self, f, t, p, *args, **kwargs):
        # A->B + A->C
        # k[0],k[1],... = k_AB, k_AC, k_B, k_C
        k, tau = self.convert_tau_k(p)
        # solutions for A, B, C in that order
        c = [np.array(f(t, 1/(k[0] + k[1])))]
        c.append(np.array(k[0]/(k[2]-k[0]-k[1]) *
                 (f(t, 1/(k[0] + k[1])) - f(t, tau[2]))))
        c.append(np.array(k[1]/(k[3]-k[0]-k[1]) *
                 (f(t, 1/(k[0] + k[1])) - f(t, tau[3]))))
        return np.array(c)

    def _kinetic_branch_two(self, f, t, p, *args, **kwargs):
        #  A->B; B->C + B->D
        #  k = [k_AB, k_BC, k_BD, k_C, k_D]
        k, tau = self.convert_tau_k(p)
        k_B = k[1] + k[2]
        a = k[0]/(k_B - k[0])
        # solutions for A, B, C, D in that order
        c = [np.array(f(t, tau[0]))]
        c.append(np.array(a*(f(t, tau[0]) - f(t, 1/(k_B)))))
        for i in (3, 4):
            c.append(np.array(a*k[i - 2]*(
                1/(k[i]-k[0])*(f(t, tau[0])-f(t, tau[i]))
                - 1/(k[i]-k_B)*(f(t, 1/(k_B)) - f(t, tau[i])))))
        return np.array(c)

    def _kinetic_branch_three(self, f, t, p, *args, **kwargs):
        # A->B + A->C; B->D + C->D
        #  k = [k_AB, k_AC, k_BD, k_CD, k_D]
        k, tau = self.convert_tau_k(p)
        # first three comp. same as branched model one
        comps = self._kinetic_branch_one(f, t, p)
        # abbreviations
        k_A = k[0] + k[1]
        c = k[2]*k[0]/(k_A - k[2])
        d = k[3]*k[1]/(k_A - k[3])
        a = -c - d
        u = a/(k_A - k[4]) + c/(k[2] - k[4]) + d/(k[3] - k[4])
        # solution for D(t)
        d_t = (a/(k[4] - k_A) * f(t, 1/k_A) +
               c/(k[4] - k[2]) * f(t, tau[2]) +
               d/(k[4] - k[3]) * f(t, tau[3]) +
               u * f(t, tau[4]))
        return np.concatenate((comps, np.array(d_t)[np.newaxis, :]))

    def _kinetic_branch_four(self, f, t, p, *args, **kwargs):
        # A->B; B->C + B->D; D->E + C->E
        #  k = [k_AB, k_BC, k_BD, k_CE, k_DE, k_E]
        k, tau = self.convert_tau_k(p)
        # first four comp. same as branched model two
        comps = self._kinetic_branch_two(f, t, p)
        # abbreviations
        k_B = k[1] + k[2]
        a = k[0]/(k_B - k[0])
        c = k[1] * k[3] / (k[3] - k[0]) + k[2] * k[4] / (k[4] - k[0])
        d = k[1] * k[3] / (k_B - k[3]) + k[2] * k[4] / (k_B - k[4])
        h = k[1] * k[3] * (k[0] - k_B) / ((k[0] - k[3]) * (k[3] - k_B))
        m = k[2] * k[4] * (k[0] - k_B) / ((k[0] - k[4]) * (k[4] - k_B))
        u = -a * (c / (k[5] - k[0]) +
                  d / (k[5] - k_B) +
                  h / (k[5] - k[3]) +
                  m / (k[5] - k[4]))
        # solution for E(t)
        e_t = (a * (c / (k[5] - k[0]) * f(t, tau[0]) +
                    d / (k[5] - k_B) * f(t, 1/k_B) +
                    h / (k[5] - k[3]) * f(t, tau[3]) +
                    m / (k[5] - k[4]) * f(t, tau[4])) +
               u * f(t, tau[5]))
        return np.concatenate((comps, np.array(e_t)[np.newaxis, :]))

    def _kinetic_branch_five(self, f, t, p, *args, **kwargs):
        # A->B; A->C + B->C
        #  k = [k_AB, k_AC, k_BC, k_C]
        k, tau = self.convert_tau_k(p)
        k_A = k[0] + k[1]
        # solutions for A, B, C in that order
        c = [np.array(f(t, 1/k_A))]
        c.append(np.array(k[0]/(k[2] - k_A)*(f(t, 1/k_A) - f(t, tau[2]))))
        A_1 = (k[0]*k[2] + k[2]*k[1] - k[1]*k_A)/((k_A - k[2]) * (k_A - k[3]))
        A_2 = k[0]*k[2]/((k_A - k[2]) * (k[2] - k[3]))
        c.append(
            np.array(A_1*(f(t, 1/k_A) - f(t, tau[3]))
                     - A_2*(f(t, tau[2]) - f(t, tau[3]))))
        return np.array(c)

    def _kinetic_branch_six(self, f, t, p, *args, **kwargs):
        # A->B; B->C; C->D + C->E
        # k = [k_AB,k_BC,k_CD,k_CE,k_D,k_E]
        k, tau = self.convert_tau_k(p)
        k_AB = k[0]
        k_BC = k[1]
        k_CD = k[2]
        k_CE = k[3]
        k_D = k[4]
        k_E = k[5]
        k_C = k_CD + k_CE
        b = k_BC * k_AB / ((k_BC - k_AB) * (k_C - k_AB))
        d = k_BC * k_AB / ((k_BC - k_AB) * (k_BC - k_C))
        g = k_BC * k_AB / (k_BC - k_AB) * (1 / (k_C - k_BC) - 1 / (k_C - k_AB))
        #
        c_A_B = self._kinetic_chain(f, t, p, num_comp=2)
        c = [c_A_B[i, :] for i in range(2)]
        c.append(np.array(b * f(t, tau[0]) +
                 d * f(t, tau[1]) + g * f(t, 1/k_C)))
        i = 0
        a = k_CD
        k = k_D
        c.append(np.array(a * (b / (k - k_AB) * f(t, tau[0]) +
                               d / (k - k_BC) * f(t, tau[1]) +
                               g / (k - k_C) * f(t, 1/k_C))
                          - a * (b / (k - k_AB) +
                                 d / (k - k_BC) +
                                 g / (k - k_C)) * f(t, tau[4 + i])))
        i += 1
        a = k_CE
        k = k_E
        c.append(np.array(a * (b / (k - k_AB) * f(t, tau[0]) +
                               d / (k - k_BC) * f(t, tau[1]) +
                               g / (k - k_C) * f(t, 1/k_C))
                          - a * (b / (k - k_AB) +
                                 d / (k - k_BC) +
                                 g / (k - k_C)) * f(t, tau[4 + i])))
        return np.array(c)

    def _kinetic_branch_seven(self, f, t, p, *args, inf_comp=None, **kwargs):
        k, tau = self.convert_tau_k(p)
        c = self._kinetic_chain(f, t, p, num_comp=3, inf_comp=inf_comp)
        comp_D = f(t, tau[-1])
        return np.concatenate((c, np.array(comp_D)[np.newaxis, :]))
    
    
    def _kinetic_branch_eight(self, f, t, p, *args, inf_comp=None, **kwargs):
        # A->C, B->C; C->D; D->E
        # k = [k_AC, k_BC, k_CD, k_DE]
        k, tau = self.convert_tau_k(p)
        # Initial population distribution between A and B defined by
        # parameter q. preliminarily fixed to 0.5 (equal distribution)
        q = 0.5
        # A(t) and B(t)
        c = [np.array(q * f(t, p['tau_1'])),
             np.array((1-q) * f(t, p['tau_2']))]
        # C(t)
        c.append(np.array(
            k[0] * q / (k[2]-k[0]) * (f(t, tau[0]) - f(t, tau[2]))
            + k[1] * (1-q) / (k[2]-k[1]) * (f(t, tau[1]) - f(t, tau[2]))))
        # D(t)
        c.append(np.array(
            k[2] * k[0] * q / (k[2]-k[0]) * (
                1 / (k[3]-k[0]) * (f(t, tau[0])-f(t, tau[3]))
                - 1 / (k[3]-k[2]) * (f(t, tau[2])-f(t, tau[3])))
            + k[2] * k[1] * (1-q) / (k[2]-k[1]) * (
                1 / (k[3]-k[1]) * (f(t, tau[1])-f(t, tau[3]))
                - 1 / (k[3]-k[2]) * (f(t, tau[2])-f(t, tau[3])))))
        # E(t)
        c.append(np.array(k[3] * k[2] * (
             k[0] * q / ((k[2]-k[0]) * (k[3]-k[0]) * (k[4]-k[0]))
                 * (f(t, tau[0])-f(t, tau[4]))
             + k[1] * (1-q) / ((k[2]-k[1]) * (k[3]-k[1]) * (k[4]-k[1]))
                 * (f(t, tau[1])-f(t, tau[4]))
             - (k[0] * q / ((k[4]-k[2]) * (k[2]-k[0]) * (k[3]-k[2]))
                + k[1] * (1-q) / ((k[4]-k[2]) * (k[2]-k[1]) * (k[3]-k[2])))
                     * (f(t, tau[2])-f(t, tau[4]))
             + 1 / (k[4]-k[3]) * (
                 k[0] *  q / ((k[2]-k[0]) * (k[3]-k[0]))
                 + k[1] * (1-q) / ((k[2]-k[1]) * (k[3]-k[1]))
                 - k[0] * q / ((k[2]-k[0]) * (k[3]-k[2]))
                 - k[1] * (1-q) / ((k[2]-k[1]) * (k[3]-k[2])))
                     * (f(t, tau[4])-f(t, tau[3])))))
        return np.array(c)

    
    def _kinetic_branch_nine(self, f, t, p, *args, inf_comp=None, **kwargs):
        k, tau = self.convert_tau_k(p)
        c = self._kinetic_chain(f, t, p, num_comp=4, inf_comp=inf_comp)
        comp_D = f(t, tau[-1])
        return np.concatenate((c, np.array(comp_D)[np.newaxis, :]))
    
    def _kinetic_branch_ten(self, f, t, p, *args, inf_comp=None, **kwargs):
        k, tau = self.convert_tau_k(p)
        c = self._kinetic_chain(f, t, p, num_comp=5, inf_comp=inf_comp)
        comp_D = f(t, tau[-1])
        return np.concatenate((c, np.array(comp_D)[np.newaxis, :]))
    
    def _kinetic_branch_eleven(self, f, t, p, *args, inf_comp=None, **kwargs):
        # A->B + A->D; B->C
        # k[0],k[1],... = k_AB, k_AD, k_BC, k_C, k_D
        # k_A: k[0] + k[1]
        # "AB", "AD", "BC", "C", "D"
        k, tau = self.convert_tau_k(p)
        k_A = k[0] + k[1]
        # solutions
        # A
        c = [np.array(f(t, 1/(k[0] + k[1])))]
        # B
        c.append(np.array(k[0]/(k[2]-k[0]-k[1]) *
                 (f(t, 1/(k[0] + k[1])) - f(t, tau[2]))))
        # C
        c.append(np.array(((k[2] * k[0] * f(t, tau[3]) * (
                                (k[3]+k[2]) * f(t, 1/(k[3]-k_A))
                                + (k_A-k[3]) * f(t, 1/(k[3]+k[2]))))
                            / ((k[3]+k[2]) * (k[3]-k_A) * (k[2]-k_A)))
                        - (((k[2] * k[0]) * (k[2]+k_A))
                            / ((k[3]+k[2]) * (k[3]-k_A)**2))))
        # D
        c.append(np.array(k[1]/(k[4]-k[0]-k[1]) *
                 (f(t, 1/(k[0] + k[1])) - f(t, tau[4]))))
        return np.array(c)
    
    
# multi start fit wrapper functions:
# run fit multiple times varying the guesses stochastically and
# return results from fit with lowest lsq

# wrapper for scipy (not used currently, but may be in the future)
def multistart_scipy(solver, fit_function, p0, num_runs=10,
                     variance=lambda mu: mu * 0.1, **solver_kwargs):
    results = []
    guesses = []
    lsq = []
    for i in range(num_runs):
        p0_run = []
        for guess in p0:
            p0_run.append(random.gauss(guess, variance(guess)))
        guesses.append(p0_run)
        results.append(solver(fit_function, p0_run, **solver_kwargs))
        lsq.append(results[i].fun)
    ind_of_minimum = lsq.index(np.min(lsq))
    return results[ind_of_minimum], guesses[ind_of_minimum]


# wrapper for lmfit (currently used)
def multistart_lmfit(fit_function, params, num_runs=10,
                     variance=lambda mu: mu * 0.1, **minimizer_kwargs):
    results = []
    guesses = []
    lsq = []
    minimizer_obj = []
    for i in range(num_runs):
        run_para = lmfit.Parameters()
        run_guesses = []
        for p in params.values():
            guess = random.gauss(p.value, variance(p.value))
            run_para.add(p.name, value=guess,
                         min=p.min, max=p.max)
            run_guesses.append(guess)
        guesses.append(run_guesses)
        minimizer_obj.append(lmfit.Minimizer(fit_function, run_para))
        try:
            results.append(minimizer_obj[i].minimize(**minimizer_kwargs))
        except Exception:
            continue
        lsq.append(np.sum([r**2 for r in results[i].residual]))
    min_ind = np.argmin(lsq)
    return results[min_ind], guesses[min_ind], minimizer_obj[min_ind]


# %%
""" global analysis """
# class for two-dimensional Global fits (Global and target analysis).
# Note: could be extended to more dimensions
class GlobalFit():
    def __init__(self, y, z,  # number_of_components,
                 x=None, model='parallel2inf2model13',
                 constraints={'min_para_ratio': None,
                              'max_amplitude_ratio': None,
                              'total_population': None},
                 enable_nonl_constr=True,
                 constant_component=False,
                 irf_sigma=None,
                 bounds='auto',
                 algorithm='SLSQP',
                 t0=0,
                 guesses='auto',
                 fun_tol=1e4, num_function_eval=1e5,
                 multistart=False, multistart_variance=0.1,
                 fixed_para=None):
        # initialize children and set default attributes
        self.error = "unknown"
        self.model_obj = FitModels()
        self.iter_count = 0
        self.ci = None
        self.number_of_decays = 2
        self.number_of_species = 2
        self.constant_component = False
        self.x = x
        # read inputs
        # blind copy
        self.y = np.array(y)
        self.z = np.ma.array(z, mask=np.isnan(z))
        self.sigma = irf_sigma
        self.bounds = bounds
        self.multistart = multistart   # either False or number of runs
        self.multistart_variance = multistart_variance
        self.t0 = t0
        self.fun_tol = fun_tol
        self.num_function_eval = num_function_eval
        self.constraints = constraints
        self.enable_nonl_constr = enable_nonl_constr
        self.fixed_para = fixed_para
        self.algorithm = algorithm
        # attributes based on inputs:
        # model, guesses, bounds
        if not self.set_model(model=model):
            self.error = 'Model not recognized.'
            return
        self.set_guesses(guesses=guesses)
        self.set_bounds(bounds=bounds)
        self.error = None

    def set_guesses(self, guesses=None):
        # reads guesses (if provided) and sets initial parameter values
        # input guesses (if provided) need to be list or 'auto'
        if guesses is None:
            try:
                guesses = self.guesses
            except Exception:
                guesses = 'auto'
        elif type(guesses) is list:
            self.guesses = guesses
        elif guesses == 'auto':
            self.guesses = self.auto_guess()
        try:
            self._set_para_values()
        except Exception:
            try:
                self.set_model()
                self._set_para_values()
            except Exception as e:
                self.guesses = None
                self.error = "Error reading guesses: " + str(e)

    def _set_para_values(self):
        # secondary function. sets initial values for lmfit parameters
        if type(self.guesses) is dict:
            for p, val in self.guesses.items():
                self.params[p].value = val
        else:
            for i, p in enumerate(self.params):
                try:
                    self.params[p].value = self.guesses[i]
                except Exception:
                    return
            try:
                self.params['t0'] = self.guesses[i + 1]
                self.params['sigma_1'] = self.guesses[i + 2]
            except Exception:
                return

    def set_model(self, model=None, inf_comp=None):
        # reads model, either parallel or kinetic (= sequential)
        # model string syntax (example in brackets):
        #   prefix: Parall or kinetic (Parall)
        #   number of decays (2)
        #   infinite component and which one (inf2)
        #   model number (model13)
        #   select components (select1+2)
        #   complete example string: Parall2inf2model13select1+2
        #   Note: this is planned to be changed to object based communication
        if model is None:
            model = self.model
        if inf_comp is None:
            inf_comp = self.constant_component
        else:
            self.constant_component = inf_comp
        # read number of decays from model string
        try:
            self.number_of_decays = int(re.findall("\d+", model)[0])
        except Exception:
            pass
        else:
            self.number_of_species = self.number_of_decays

        if re.search('paral', model, re.I):
            try:
                no = int(re.findall('(?<=model)\d+', model)[-1])
            except Exception:
                print('Model number not readable. Using Chain.')
                self.model_obj.set_model('parallel')
            else:
                self.model_obj.set_model(no)
        elif re.search('kinetic|seque', model, re.I):
            # read kinetic model number
            try:
                no = int(re.findall('(?<=model)\d+', model)[-1])
            except Exception:
                print('Model number not readable. Using Chain.')
                self.model_obj.set_model('chain')
            else:
                self.model_obj.set_model(no)
            # overwrite number of decays and species
            # if dictated by specific model
            try:
                self.number_of_decays = self.model_obj.get_model()[
                    'properties']['number_of_decays']
                self.number_of_species = self.model_obj.get_model()[
                    'properties']['number_of_species']
            except Exception:
                pass
        else:
            return False
        # sets invisible components based on model,
        # i.e. parts of a sequential model that do not
        # contribute to the data.
        self._invisible_comps = []
        if re.search("select", model, re.I):
            self._visible_comps = []
            select = re.findall("(?<=select).*", model)[0].split('+')
            for i in range(self.number_of_species):
                if not str(i + 1) in select:
                    self._invisible_comps.append(i)
                else:
                    self._visible_comps.append(i)
        else:
            self._visible_comps = [i for i in range(self.number_of_species)]
        self.constant_component = bool(re.search('inf', model))
        self._init_params()
        self.model = model
        return True

    def _init_params(self, values=None, fixed_para=None):
        # secondary function initializing lmfit parameters based on model
        if fixed_para is None:
            fixed_para = {}
        if self.constant_component:
            fixed_para['tau_' + str(self.number_of_decays)] = np.inf
        if not self.sigma:
            fixed_para['t0'] = self.t0
        self.params = lmfit.Parameters()
        for i in range(self.number_of_decays):
            self.params.add('tau_' + str(i + 1), value=1)
        self.params.add('t0', value=self.t0)
        if self.sigma:
            self.params.add('sigma_1', value=self.sigma)
        if values is not None:
            self.set_guesses(values)
        for key, val in fixed_para.items():
            self.params[key].vary = False
            self.params[key].value = val

    def run_fit(self, fun_tol=None, num_function_eval=None,
                queue=None, enable_nonl_constr=None, fixed_para=None,
                guesses=None, bounds=None, model=None,
                constant_component=None):
        # main method, executing global fit
        # kwargs are only to overwrite previously set properties of the class
        self.error = "unknown"
        if self.number_of_decays <= 0:
            self.ycomps, self.x_comps, self.fit_matrix = (
                self.calc_results([np.inf]))
            self.error = None
            self.result = BlankObject()
            self.result.message = ("All parameters have been fixed. "
                                   + "No fit executed.")
            self.result.nfev = 0
            return

        if fun_tol is None:
            fun_tol = self.fun_tol
        if num_function_eval is None:
            num_function_eval = self.num_function_eval
        if enable_nonl_constr is None:
            enable_nonl_constr = self.enable_nonl_constr
        else:
            self.enable_nonl_constr = enable_nonl_constr
        if fixed_para is None:
            fixed_para = self.fixed_para
        if model is not None or constant_component is not None:
            self.set_model(model=model, inf_comp=constant_component)

        self.iter_count = 0

        minimizer_kwargs = {'method': self.algorithm}
        if not self.algorithm.lower() in ['basinhopping']:
            minimizer_kwargs['max_nfev'] = num_function_eval
        if self.algorithm.lower() in ['least_squares', 'leastsq',
                                      'differential_evolution']:
            minimizer_kwargs['ftol'] = fun_tol
        if bounds is not None:
            self.set_bounds(bounds=bounds)
        if guesses is not None:
            self.set_guesses(guesses=guesses)
        try:
            if self.multistart:
                self.result, p0_multistart, minimizer = multistart_lmfit(
                    lambda p: self.calculate_resid(p, queue),
                    self.params, **minimizer_kwargs,
                    num_runs=self.multistart,
                    variance=lambda mu: mu*self.multistart_variance)
            else:
                minimizer = lmfit.Minimizer(
                    lambda p: self.calculate_resid(p, queue), self.params)
                self.result = minimizer.minimize(**minimizer_kwargs)
        except Exception as e:
            self.error = str(e)
        else:
            para_values = []
            for p in self.result.params.values():
                para_values.append(p)
            try:
                self.ci = [
                    self.result.params[p].stderr for p in self.result.params]
                if self.ci[0] is None:
                    self.ci = None
            except Exception as e:
                print(e)
                self.ci = None
            # result processing
            self.ycomps, self.x_comps, self.fit_matrix = self.calc_results(
                self.result.params)
            self.fit_matrix = np.transpose(self.fit_matrix)
            self.error = None

    def fit_report(self):
        # returns lmfit fit_report or branch report if not successful
        if self.error is not None:
            return ['Error during fit or object initialization:', self.error]
        try:
            return lmfit.fit_report(self.result)
        except Exception:
            pass
        report = []
        try:
            if not self.result.success:
                report.append('Fit did not converge successfully')
        except Exception:
            pass
        report.append(self.result.message)
        report.append('Number of function evaluations: ' +
                      str(self.result.nfev))
        return report

    def calculate_resid(self, p, queue=None):
        # fit residual (simple subtraction)
        try:
            y, x, fit_mat = self.calc_results(p)
        except Exception as e:
            print('Exception in GlobalFit.calculate_resid:\n' +
                  str(e))
        else:
            resid = np.transpose(self.z) - fit_mat
        self.iter_count += 1
        if queue is not None:
            queue.put({'conc': y, 'num_iter': self.iter_count})
        return resid

    def calc_results(self, p):
        # both used during optimization and to obtain global fit
        # parameters afterwards.
        # calculate y components according to utilized model
        y = self.model_obj.fit_function(
            self.y, p, num_comp=self.number_of_decays, sigma=self.sigma)
        # calculate x components via matrix inversion
        try:
            x = np.dot(np.transpose(self.z), np.linalg.pinv(y))
        except Exception as e:
            try:
                x = np.dot(self.z, np.linalg.pinv(y))
            except Exception as ex:
                print(e, ex)
                x = np.zeros((np.shape(self.z)[1], self.number_of_species))
        for inv in self._invisible_comps:
            x[:, inv] = np.zeros((np.shape(self.z)[1]))
        # calculate fit matrix (fit_mat) for residual
        fit_mat = np.dot(np.asarray(x)[:, self._visible_comps], np.asarray(y)[
                        self._visible_comps, :])
        return y, x, fit_mat

    # auto guess: very simplistic, should be improved!
    def auto_guess(self):
        if self.number_of_decays <= 0:
            return np.inf
        p0 = [1 for i in range(self.number_of_decays)]
        p0[-1] = self.y[-1]/1.5
        for i in range(1, self.number_of_decays - 1):
            p0[i] = p0[i - 1] + (p0[-1] - p0[0])/(self.number_of_decays - 1)
        if self.constant_component:
            p0[-1] = np.inf
        return p0

    # parameter bounds & constraints
    def set_bounds(self, bounds=None):
        if bounds is None:
            bounds = self.bounds
        if bounds == 'auto':
            if self.algorithm.lower() in ['differential_evolution']:
                upper = 1e13
            else:
                upper = np.inf
            for p in self.params:
                self.params[p].min = 1e-12
                self.params[p].max = upper
            if 't0' in self.params:
                self.params['t0'].min = -upper
        elif bounds is None:
            return
        else:
            for i, p in enumerate(self.params):
                self.params[p].min = bounds[i]
                self.params[p].max = bounds[self.number_of_decays + i]
        self.bounds = bounds
# %%
""" line fit """


class LineFitParaDicts():
    # helper dictionary and base class for class LineFit
    # used to access parameter names, dimensions, standard bounds and other
    # properties of parameters from the line fit models
    def __init__(self):
        self._para_name_dict = {'gauss': ['amp', 'x0', 'sigma'],
                                'lorentz': ['amp', 'x0', 'gamma'],
                                'voigt': ['amp', 'x0', 'sigma', 'gamma'],
                                'exp': ['tau', 'amp'],
                                'kineticexp': ['tau', 'amp_1'],
                                'cos': ['amp', 'omega', 'phi'],
                                'osc': ['omega', 'phi'],
                                'damp': ['amp', 'tau'],
                                'linear': ['a'],
                                'poly': ['a'],
                                'const': ['amp'],
                                'hermite': ['amp']}
        self._shared_paras = ['x0', 'sigma', 'spacing', 'wdfactor']
        self._para_axis_dict = {'x0': 'x',
                                'sigma': 'x2',
                                'gamma': 'x2',
                                'const': 'y',
                                'amp': 'y',
                                'tau': 'x',
                                'omega': 'x',
                                'phi': 'x2',
                                'a': 'y',
                                'spacing': 'x',
                                'wdfactor': 'x2'}
        self._name_dict = {'exp': "Exponential",
                           'kineticexp': "Exponential",
                           'damp': "Damping",
                           'gauss': "Gaussian",
                           'gauss_square': "Gaussian Squared",
                           'lorentz': "Lorentzian",
                           'voigt': "Voigt",
                           'cos': "Cosine",
                           'osc': "Cosine",
                           'poly': "Polynomial",
                           'linear': "Linear",
                           'const': "Offset",
                           'x0': "x0",
                           'sigma': "Sigma",
                           'spacing': "Spacing",
                           'wdfactor': "Width Factor",
                           'hermite': "Hermite Polynomial"}
        self._std_bds = {
            'exp': {'tau': [1e-24, np.inf], 'sigma': [0, np.inf]},
            'kineticexp': {'tau': [1e-24, np.inf], 'sigma': [0, np.inf]},
            'damp': {'tau': [1e-24, np.inf], 'sigma': [0, np.inf]},
            'cos': {'omega': [0, np.inf],
                    'phi': [-np.pi, np.pi]},
            'osc': {'omega': [0, np.inf],
                    'phi': [-np.pi, np.pi]},
            'gauss': {'x0': [-np.inf, np.inf], 'sigma': [0, np.inf]},
            'lorentz': {'gamma': [0, np.inf]},
            'hermite': {'sigma': [1e-24, np.inf]}}
        self._std_bds['gauss_square'] = self._std_bds['gauss']
        for key, val in self._para_name_dict.items():
            self._para_name_dict[key] = {'individual': val}
            try:
                for i, name in enumerate(val):
                    self._para_name_dict[key]['individual'][i] = '_'.join(
                        [key, name])
            except Exception:
                self._para_name_dict[key]['individual'] = [key]
        for key in ('exp', 'kineticexp', 'damp'):
            self._para_name_dict[key]['shared'] = ['x0_1']
        for key in ['hermite']:
            self._para_name_dict[key]['shared'] = ['x0_1', 'sigma_1']
        # certain models require constraints normally, but this can disabled
        # by setting the following to False:
        self.enable_default_constraints = True
        self._default_constraints = {'kineticexp': {'mode': 'factor',
                                                    'ref': 'preceding',
                                                    'paras': ['tau'],
                                                    'min': [0.5],
                                                    'max': [np.inf]}}

    def get_attributes(self):
        return self._name_dict, self._para_axis_dict, self._shared_paras

    def get_reverse_para_dict(self):
        dct = {}
        for key, values in self._para_name_dict.items():
            for val in values.values():
                dct[val] = key
        return dct


# %%
class LineFit(LineFitParaDicts):
    def __init__(self, x=None, y=None, dy=None, model='exp', fixed_para=None,
                 para_correl=None,
                 method='leastsq', num_function_eval=int(1e4),
                 fun_tol=1e-2, guesses=None, bounds=None,
                 max_fun_degree=10, x0=0,
                 fit_range=None,
                 reject_outliers=False,
                 outlier_threshold=2,
                 outlier_reject_maxiter=20):
        LineFitParaDicts.__init__(self)
        self.model_obj = FitModels()
        self._submodels = list(self.model_obj.model_dict.keys())
        # read inputs (blind copy)
        self.x = x
        self.y = y
        self.dy = dy
        self.fit_range = fit_range
        self.model = model
        self.max_fun_degree = max_fun_degree
        self.method = method
        self.num_function_eval = num_function_eval
        self.fun_tol = fun_tol
        self.reject_outliers = reject_outliers
        self.outliers = [[], []]
        self.non_outliers = []
        self.forced_outliers = []
        self.outlier_threshold = outlier_threshold
        self.outlier_reject_maxiter = outlier_reject_maxiter
        # read inputs (conditional)
        if fixed_para is None:
            self.fixed_para = {}
        else:
            self.fixed_para = fixed_para
        if para_correl is None:
            self.para_correl = {}
        else:
            self.para_correl = para_correl
        try:
            self.read_model()
        except Exception:
            self.model = 'exp'
            self.read_model()
            print('Invalid model for class LineFit. Using exp.')
        self.init_parameters()
        if guesses is not None:
            self.set_guesses(guesses)
        if bounds is not None:
            self.set_bounds(bounds)
        else:
            self.set_std_bounds()

    def set_attributes(self, update=True, **kwargs):
        for key, val in kwargs.items():
            try:
                exec("self." + key)
            except Exception:
                try:
                    exec("self._" + key)
                except Exception as e:
                    print("Error setting fit object attributes: " + str(e))
                else:
                    exec("self._" + key + " = val")
            else:
                exec("self." + key + " = val")
        if 'model' in kwargs.keys() or 'selected_comp' in kwargs.keys():
            self.read_model()
            self.init_parameters()
            if 'bounds' in kwargs.keys():
                self.set_bounds(kwargs['bounds'])
            else:
                self.set_std_bounds()
        elif 'bounds' in kwargs.keys():
            self.set_bounds(kwargs['bounds'])
        if 'guesses' in kwargs.keys():
            self.set_guesses(kwargs['guesses'])

        if 'fixed_para' in kwargs.keys():
            self.set_fixed_para()

    def run_fit(self, x=None, y=None, dy=None, model=None, fixed_para=None,
                para_correl=None, method=None, num_function_eval=None,
                fun_tol=None, guesses=None, bounds=None, queue=None,
                fit_range=None, reject_outliers=None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if dy is not None:
            self.dy = dy
        if fit_range is not None:
            self.fit_range = fit_range
        if reject_outliers is None:
            reject_outliers = self.reject_outliers
        if self.fit_range is not None:
            try:
                ind = slice(np.where(self.x >= self.fit_range[0])[0][0],
                            np.where(self.x <= self.fit_range[1])[0][1] + 1)
            except Exception:
                pass
            else:
                self.x = self.x[ind]
                self.y = self.y[ind]
                try:
                    self.dy = self.dy[ind]
                except Exception:
                    pass
                self.fit_range_slice = ind

        if model is None:
            model = self.model
        if not self.get_fit_function(model=model):
            return False, "Fit model not recognized."
        if para_correl is not None:
            self.set_para_correl(para_correl)
        if method is None:
            method = self.method
        if num_function_eval is None:
            num_function_eval = self.num_function_eval
        if fun_tol is None:
            fun_tol = self.fun_tol

        # set up fit function, parameters, guesses and bounds
        if guesses is not None:
            self.set_guesses(guesses)
        if bounds == 'auto':
            self.set_std_bounds()
        elif bounds is not None:
            self.set_bounds(bounds)
        self.set_fixed_para(fixed_para)

        # set up minimizer incl. options
        minimizer = lmfit.Minimizer(
            lambda p: self.calc_residual(p, queue=queue), self.params)
        minimizer_kwargs = {'method': method,
                            'max_nfev': num_function_eval}
        if method.lower() in ['least_squares', 'leastsq',
                              'differential_evolution']:
            minimizer_kwargs['ftol'] = fun_tol
        # run fit
        try:
            self.minimize(minimizer, reject_outliers=reject_outliers,
                          **minimizer_kwargs)
        except Exception as e:
            self.covar = None
            self.stderrors = None
            self.result = None
            try:
                self.curve = self.fit_function(self.params)(self.x)
            except Exception:
                self.curve = None
                self.residual = None
            else:
                self.residual = self.calc_residual(self.params)
            msg = split_string_into_lines(str(e), max_len=90)
            # temp
            raise
            return False, "Exception during fit: " + msg
        else:
            # results processing
            self.covar = minimizer.covar
            self.residual = self.calc_residual(self.result.params)
            try:
                self.stderrors = {}
                for key, p in self.result.params.items():
                    self.stderrors[key] = p.stderr
                    if p.stderr is None:
                        self.stderrors = None
                        break
            except Exception as e:
                print(e)
                self.stderrors = None
            self.model = model
            self.method = method
            self.num_function_eval = num_function_eval
            try:
                report = lmfit.fit_report(self.result)
            except Exception:
                report = "Failure writing fit report"
            return True, report

    def minimize(self, minimizer, reject_outliers=False, multistart=False,
                 **minimizer_kwargs):
        def outlier_rejection(x, y, thresh, fit):
            dev = np.abs(y - fit)
            try:
                dev = dev / np.median(dev)
            except Exception:
                return x, y, []
            else:
                outlier_inds = np.logical_and(
                    dev >= thresh, np.isin(x, self.non_outliers, invert=True))
                non_outlier_inds = np.invert(outlier_inds)
                return x[non_outlier_inds], y[non_outlier_inds], outlier_inds
        if reject_outliers:
            x_backup = self.x
            y_backup = self.y
            self.result = minimizer.minimize(**minimizer_kwargs)
            forced_inds = np.isin(self.x, self.forced_outliers, invert=True)
            self.x = self.x[forced_inds]
            self.y = self.y[forced_inds]
            self.curve = self.fit_function(self.result.params)(self.x)
            x, y, outlier_inds = outlier_rejection(self.x, self.y,
                                                   self.outlier_threshold,
                                                   self.curve)
            k = 0
            while np.any(outlier_inds) and k <= self.outlier_reject_maxiter:
                self.outliers[0].extend(self.x[outlier_inds])
                self.outliers[1].extend(self.y[outlier_inds])
                self.x = x
                self.y = y
                self.result = minimizer.minimize(**minimizer_kwargs)
                curve = self.fit_function(self.result.params)(self.x)
                x, y, outlier_inds = outlier_rejection(self.x, self.y,
                                                       self.outlier_threshold,
                                                       curve)
                k += 1
            self.x = x_backup
            self.y = y_backup
            self.curve = self.fit_function(self.result.params)(self.x)
        else:
            self.result = minimizer.minimize(**minimizer_kwargs)
            self.curve = self.fit_function(self.result.params)(self.x)

    def get_para_dict(self):
        return self._para_name_dict, self._shared_para_dict

    def set_init_to_result(self, variance=0.0):
        def vary_init(val):
            return random.gauss(val, variance*val)
        if variance:
            val_func = vary_init
        else:
            def val_func(val): return val
        try:
            for p in self.result.params:
                if p in self.params:
                    self.params[p].value = val_func(
                        self.result.params[p].value)
        except AttributeError:
            pass
        except Exception:
            raise

    def set_guesses(self, guesses, model=None):
        if model is None:
            model = self.model
        try:
            self.params.keys()
        except Exception:
            self.init_parameters(model=model)
        if guesses == 'auto':
            if model.startswith(('exp', 'kin', 'cos', 'osc')):
                self.auto_guess_exp_func()
            elif re.search('gauss', model, re.I):
                self.auto_guess_gaussian()
        else:
            try:
                for key, val in guesses.items():
                    try:
                        if self.params[key].expr is None:
                            self.params[key].set(value=val)
                        else:
                            self.params[key].set(value=None)
                    except Exception:
                        pass
            except Exception:
                pass

    def set_bounds(self, bounds, model=None):
        try:
            self.params.keys()
        except Exception:
            self.init_parameters(model=model)
        for key, bds in bounds.items():
            try:
                self.params[key].set(min=bds[0], max=bds[1])
            except Exception:
                pass

    def get_error_curves(self):
        para = {}
        for key, p in self.result.params.items():
            para[key] = p.value - p.stderr
        lower_curve = self.fit_function(para)(self.x)
        for key, p in self.result.params.items():
            para[key] = p.value + p.stderr
        upper_curve = self.fit_function(para)(self.x)
        return lower_curve, upper_curve

    def get_fit_function(self, model=None):
        if model is None:
            model = self.model
        try:
            if model.startswith(('lin', 'pol')):
                self.fit_function = self._polynomial_function
            else:
                if model.startswith('exp'):
                    fun = self._exp_fit_function
                elif model.startswith('kin'):
                    fun = self._kinetic_exp_fit_function
                    self.model_obj.set_model(self._submodel)
                elif model.startswith('gau'):
                    fun = self._gaussian_fit_function
                elif model.startswith('gas'):
                    fun = self._gaussian_sqr_fit_function
                elif re.search('hermit', model, re.I):
                    fun = self._hermite_poly_function
                elif re.search('loren', model, re.I):
                    fun = self._lorentz_fit_function
                elif re.search('voigt', model, re.I):
                    fun = self._voigt_fit_function
                else:
                    return
                self.fit_function = (
                    lambda p, *args, **kwargs: self._fit_function_wrap(
                        p, fun, *args, **kwargs))
        except Exception:
            return
        else:
            self.model = model
            return self.fit_function

    def auto_guess_gaussian(self, x=None, y=None):
        x, y, amp, xlower, xupper = self._get_data_limits(x=x, y=y)
        guesses = {}
        for key, func in self.sub_functions.items():
            n = func['degree']
            try:
                peak_inds, peak_val = find_peaks(y, height=0.1 * np.abs(amp),
                                                 distance=len(x)/(4 * n))
            except Exception:
                peak_inds = None
                peak_val = None
            paras = func['para_names']['individual']
            for para in paras:
                if re.search('const_amp', para):
                    guesses['const_amp_1'] = min(y)
                elif re.search('\_amp', para):
                    for i in range(n):
                        try:
                            guesses["_".join([para, str(i + 1)])
                                    ] = peak_val['peak_heights'][i]
                        except Exception:
                            guesses["_".join([para, str(i + 1)])
                                    ] = 0.3 * (i + 1) * amp
                elif re.search('x0', para):
                    for i in range(n):
                        try:
                            guesses["_".join([para, str(i + 1)])
                                    ] = x[peak_inds[i]]
                        except Exception:
                            guesses["_".join([para, str(i + 1)])
                                    ] = xlower + (n - i) * (xupper - xlower)
            for para in paras:
                if re.search('sigma|width', para, re.I):
                    for i in range(n):
                        try:
                            guesses["_".join([para, str(i + 1)])] = 0.01 * \
                                guesses["_".join(['gauss_x0', str(i + 1)])]
                        except Exception:
                            pass
        self.set_guesses(guesses)

    def _get_data_limits(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if self.sub_functions is None:
            self.read_model()
        try:
            self.params.keys()
        except Exception:
            self.init_parameters()
        try:
            amp = (np.max(np.ma.array(y, mask=np.isnan(y)))
                   if np.max(np.ma.array(y, mask=np.isnan(y))) >
                   np.abs(np.min(np.ma.array(y, mask=np.isnan(y))))
                   else np.min(np.ma.array(y, mask=np.isnan(y))))
        except Exception:
            amp = 10
        xupper = 0.6667*np.max(np.ma.array(x, mask=np.isnan(x)))
        if (np.max(np.ma.array(x, mask=np.isnan(x))) > 0 and
                np.min(np.ma.array(x, mask=np.isnan(x))) < 0):
            xlower = 0
        else:
            xlower = np.min(np.ma.array(x, mask=np.isnan(x)))
        xlower = xlower + 0.1*(np.abs(xlower-xupper))
        return x, y, amp, xlower, xupper

    def auto_guess_exp_func(self, x=None, y=None):
        def auto_amp(i):
            return np.exp(-0.4*(i)+0.5)*amp

        def auto_tau(i):
            return (xupper+xlower)/2*np.exp(-i)

        x, y, amp, xlower, xupper = self._get_data_limits(x=x, y=y)
        guesses = {}
        for key, func in self.sub_functions.items():
            n = func['degree']
            if key.startswith('kin'):
                guesses['exp_amp_1'] = auto_amp(1)
                paras = [p for p in func['para_names']
                         ['individual'] if p != 'exp_amp_1']
            else:
                paras = func['para_names']['individual']
            for para in paras:
                if re.search('const_amp', para):
                    guesses['const_amp_1'] = y[-1] if x[-1] > x[0] else y[0]
                elif re.search('\_amp', para):
                    for i in range(n):
                        guesses["_".join([para, str(i + 1)])] = auto_amp(i + 1)
                elif re.search('(tau)|(omega)', para):
                    for i in range(n):
                        guesses["_".join([para, str(i + 1)])] = auto_tau(n - i)
                elif re.search('phi', para):
                    for i in range(n):
                        guesses["_".join([para, str(i + 1)])] = 0
        self.set_guesses(guesses)

    def set_std_bounds(self):
        bounds = {}
        for key, func in self.sub_functions.items():
            if key in self._std_bds.keys():
                for para, std_bds in self._std_bds[key].items():
                    for i in range(func['degree']):
                        bounds['_'.join([key, para, str(i + 1)])] = std_bds
        self.set_bounds(bounds)

    def init_parameters(self, model=None, fixed_para=None, para_correl=None):
        def add_para(name, **kwargs):
            if name not in self.params:
                self.params.add(name, **kwargs)

        if model is None:
            model = self.model
        else:
            self.read_model(model)

        if self.params is None:
            self.params = lmfit.Parameters()

        for key, func in self.sub_functions.items():
            for name in func['para_names']['individual']:
                if re.search('_[1-9]+', name):
                    add_para(name, value=1)
                else:
                    for i in range(func['degree']):
                        add_para('_'.join([name, str(i + 1)]), value=1)
            try:
                for name in func['para_names']['shared']:
                    add_para(name, value=1)
            except KeyError:
                pass
            except Exception as e:
                print(e)
        if re.search('conv', self.model):
            add_para('sigma_1', value=1, min=0)
        self.model = model
        self.set_para_correl(para_correl)
        self.set_fixed_para(fixed_para)

    def read_model(self, model=None):
        if model is None:
            model = self.model
        self.sub_functions = {}
        # reading model
        # 1) functions with a degree, e.g. exponential, biexponential etc.
        # naming example: exp2poly1
        funcs = re.findall(r'([A-Za-z]+)([1-9]\d*)', model)
        # 2) functions without degree (currently only constant)
        if re.search('const', model):
            funcs.append(('const', 1))
        # 3) special affixes:
        #   model:      specific (sub-)model
        #               currently only available for kinetic exponentials
        #   select:     models with multiple correlated components of
        #               which one or more are selected AND added (if multiple)
        if re.search('model', model, re.I):
            submodel_num = re.findall('(?<=model)\d+', model)
            try:
                self._submodel = self.model_obj.get_model_name(submodel_num[0])
            except Exception:
                self._submodel = 'parallel'

        self._selected_comp = None
        try:
            select_str = re.findall(
                '(?<=select)[\d\+]+', model)[0].strip().split('+')
        except Exception:
            pass
        else:
            if len(select_str) > 1:
                self._selected_comp = [int(s) for s in select_str]
            else:
                self._selected_comp = [int(select_str[0])]

        # writing sub function dict
        if len(funcs) == 0:
            self.sub_functions[model] = {
                'degree': 1, 'para_names': self._para_name_dict[model]}
        else:
            for func, degree in funcs:
                try:
                    dct = {}
                    for key, val in self._para_name_dict[func].items():
                        dct[key] = [v for v in val]
                except Exception:
                    pass
                else:
                    self.sub_functions[func] = {'degree': int(degree),
                                                'para_names': dct}

        # (model-) specific modifications of sub functions
        if 'kineticexp' in self.sub_functions.keys():
            sf = self.sub_functions['kineticexp']
            if self._selected_comp is not None:
                sf['para_names']['individual'].pop(1)
                for i in self._selected_comp:
                    if ('kineticexp_amp_' + str(i)
                            not in sf['para_names']['individual']):
                        sf['para_names']['individual'].append(
                            'kineticexp_amp_' + str(i))
        if ('osc' in self.sub_functions.keys()
            and not ('damp' in self.sub_functions.keys()
                     or 'gauss' in self.sub_functions.keys())):
            self.sub_functions['cos'] = self.sub_functions['osc']
            self.sub_functions['cos']['para_names'] = (
                self._para_name_dict['cos'])
            del self.sub_functions['osc']
        if 'linear' in self.sub_functions.keys():
            self.sub_functions['poly'] = self.sub_functions['linear']
            self.sub_functions['poly']['degree'] = 1
            del self.sub_functions['linear']
        if 'poly' in self.sub_functions.keys():
            self.sub_functions['poly']['degree'] += 1
        # (re-)initializing parameters
        self.params = lmfit.Parameters()

    def set_fixed_para(self, fixed_para=None):
        if fixed_para is None:
            fixed_para = self.fixed_para
        else:
            self.fixed_para = fixed_para
        self._fixed_para = {}
        for key, val in fixed_para.items():
            self._fixed_para[key] = val
            if not re.search('[1-9]', key):
                for i in range(self.max_fun_degree):
                    self._fixed_para['_'.join([key, str(i + 1)])] = val
        for p in self.params:
            if p in self._fixed_para.keys():
                self.params[p].value = self._fixed_para[p]
                self.params[p].vary = False
            else:
                self.params[p].vary = True
        return self._fixed_para

    def set_para_correl(self, para_correl=None):
        if para_correl is not None:
            self.para_correl = para_correl
        for p in self.params:
            if p in self.para_correl.keys():
                self.params[p].expr = self.para_correl[p]
            else:
                self.params[p].expr = None

    def set_default_constraints(self):
        for sf in self.sub_functions.keys():
            if sf in self._default_constraints.keys():
                if self._default_constraints[sf]['ref'] == 'preceding':
                    for j, para in enumerate(
                            self._default_constraints[sf]['paras']):
                        paraname = "_".join([sf, para, ""])
                        mi = self._default_constraints[sf]['min'][j]
                        ma = self._default_constraints[sf]['max'][j]
                        i = 2
                        while True:
                            try:
                                self.params[paraname + str(i)]
                            except Exception:
                                break
                            else:
                                self.set_ineq_constraint(
                                    paraname + str(i - 1),
                                    {paraname + str(i): {
                                        'min': mi, 'max': ma}},
                                    mode=self._default_constraints[sf]['mode'])
                                i += 1
                elif self._default_constraints[sf]['ref'] == 'first':
                    for para in enumerate(
                            self._default_constraints[sf]['paras']):
                        paraname = "_".join([sf, para, ""])
                        mi = self._default_constraints[sf]['min'][j]
                        ma = self._default_constraints[sf]['max'][j]
                        ref = paraname + "1"
                        i = 2
                        while True:
                            try:
                                self.params[paraname + str(i)]
                            except Exception:
                                break
                            else:
                                self.set_ineq_constraint(
                                    ref,
                                    {paraname + str(i): {
                                        'min': mi, 'max': ma}},
                                    mode=self._default_constraints[sf]['mode'])
                                i += 1

    def set_ineq_constraint(self, para_ref, paras, mode='delta'):
        # para_ref reference parameter name
        # paras: dict of form {name: {min: val, max: val}}
        if mode == 'delta':
            for name, constr in paras.items():
                self.params.add('_minus_'.join([name, para_ref]),
                                vary=self.params[name].vary,
                                value=self.params[name]-self.params[para_ref],
                                **constr)
                self.params[name].expr = "-".join(
                    ['_minus_'.join([name, para_ref]), para_ref])
        elif mode == 'factor':
            for name, constr in paras.items():
                try:
                    val = self.params[name]/self.params[para_ref]
                except Exception:
                    try:
                        val = constr['min']
                    except Exception:
                        try:
                            val = constr['max']
                        except Exception:
                            val = 0
                self.params.add('_div_'.join([name, para_ref]),
                                vary=self.params[name].vary,
                                value=val, **constr)
                self.params[name].expr = "*".join(
                    ['_div_'.join([name, para_ref]), para_ref])
        else:
            print('Invalid inequality constraint mode.')

    def _kinetic_exp_fit_function(self, x, p, *args, return_comps=False,
                                  **kwargs):
        num_comp = self.sub_functions['kineticexp']['degree']
        paras = {}
        for para in p:
            if re.search('tau', para, re.I):
                paras['tau_' + re.findall('\d+', para)[-1]] = p[para]
            else:
                paras[para] = p[para]
        if 'sigma_1' in p.keys():
            sigma = p['sigma_1']
        else:
            sigma = None
        components = self.model_obj.fit_function(x, paras, num_comp=num_comp,
                                                 sigma=sigma)
        c = self._selected_comp[0]
        comps = [paras['kineticexp_amp_' + str(c)]*components[c - 1]]
        for c in self._selected_comp[1:]:
            comps.append(paras['kineticexp_amp_' + str(c)]*components[c - 1])
        if 'const_amp_1' in p.keys():
            if sigma is None:
                comps.append(
                    p['const_amp_1']*exp(x, np.inf, t0=p['x0_1']))
            else:
                comps.append(
                    p['const_amp_1']*gauss_conv_exp(
                        x, np.inf, sigma, t0=p['x0_1']))
        y = np.sum(np.asarray(comps), axis=0)
        if return_comps:
            return y, comps
        else:
            return y

    def _kinetic_exp_fit_function_old(self, x, p, *args, return_comps=False,
                                      **kwargs):
        def amplitudes(j, m):
            if j == 1 and m == 1:
                return 1
            else:
                a = np.prod([1/g for g in tau[:m]])
                b = np.prod(
                    [1/tau[i] - 1/tau[j] for i in range(m + 1) if i != j])
                if b == 0:
                    return 0
                else:
                    return a/b

        deg = self.sub_functions['kineticexp']['degree']
        tau = [p['kineticexp_tau_' + str(i+1)] for i in range(deg)]
        if re.search('conv', self.model, re.I):
            f = gauss_conv_exp
            sigma = p['sigma_1']
        else:
            f = exp
            sigma = 0
        fun = np.zeros(len(x))
        comps = {}
        for j in range(deg):
            f = amplitudes(j, deg - 1)*f(x, tau[j], sigma, t0=p['x0_1'])
            fun += f
            comps['exp_' + str(j + 1)] = f
        output = p['kineticexp_amp_1']*np.sum([], axis=0)
        if 'const_1' in p.keys():
            output += p['const_1']
        if return_comps:
            return fun, comps
        else:
            return fun

    def _gaussian_fit_function(self, *args, return_comps=False, **kwargs):
        f, comps = self._line_fit_function(*args, case='gauss', **kwargs)
        if return_comps:
            return f, comps
        else:
            return f

    def _gaussian_sqr_fit_function(self, *args, return_comps=False, **kwargs):
        f, comps = self._line_fit_function(*args, case='gausquared', **kwargs)
        if return_comps:
            return f, comps
        else:
            return f

    def _lorentz_fit_function(self, *args, return_comps=False, **kwargs):
        f, comps = self._line_fit_function(*args, case='lorentz', **kwargs)
        if return_comps:
            return f, comps
        else:
            return f

    def _voigt_fit_function(self, *args, return_comps=False, **kwargs):
        f, comps = self._line_fit_function(*args, case='voigt', **kwargs)
        if return_comps:
            return f, comps
        else:
            return f

    def _line_fit_function(self, x, p, *args, case='gauss', **kwargs):
        def gaussian_fit(i):
            return p['gauss_amp_' + str(i)]*gaussian(
                x, p['gauss_x0_' + str(i)], p['gauss_sigma_' + str(i)])

        def lorentzian_fit(i):
            return p['lorentz_amp_' + str(i)]*lorentzian(
                x, p['lorentz_x0_' + str(i)], p['lorentz_gamma_' + str(i)])

        def voigt_fit(i):
            gamma = p['voigt_gamma_' + str(i)]
            s = p['voigt_sigma_' + str(i)]
            z = ((x - p['voigt_x0_' + str(i)]) + 1j*gamma)/(s*np.sqrt(2))
            return (p['voigt_amp_' + str(i)] / (s*np.sqrt(2 * np.pi))
                    * np.real(wofz(z)))

        def gaussian_square_fit(i):
            return p['gauss_amp_' + str(i)]*np.sqrt(gaussian(
                x, p['gauss_x0_' + str(i)], p['gauss_sigma_' + str(i)]))

        if re.search('gau', case, re.I):
            fname = 'gauss'
            f = gaussian_square_fit
        elif re.search('loren', case, re.I):
            fname = 'lorentz'
            f = lorentzian_fit
        elif re.search('voigt', case, re.I):
            fname = 'voigt'
            f = voigt_fit

        fun = np.zeros(len(x))
        comps = {}
        for i in range(self.sub_functions[fname]['degree']):
            comps['_'.join([fname, str(i + 1)])] = f(i + 1)
            fun += comps['_'.join([fname, str(i + 1)])]

        for key in ('osc', 'cos'):
            if key in self.sub_functions.keys():
                osc_fun = np.zeros(len(x))
                for i in range(self.sub_functions[key]['degree']):
                    f = np.cos(
                        x * p["_omega_".join([key, str(i + 1)])]
                        + p["_phi_".join([key, str(i + 1)])])
                    osc_fun += f
                    comps['osc_' + str(i + 1)] = f
                fun = fun * osc_fun
                break
        if 'const' in self.sub_functions.keys():
            f = p['const_amp_1']*np.ones(len(x))
            fun += f
            comps['const'] = f
        return fun, comps

    def _exp_fit_function(self, x, p, *args, return_comps=False, **kwargs):
        def exp_func_normal(num, sub_func='exp'):
            return p['_'.join([sub_func, 'amp', num])] * exp_func(
                x, p['_'.join([sub_func, 'tau', num])], sigma,
                t0=p['x0_1'], **kwargs)

        if re.search('conv', self.model):
            exp_func = gauss_conv_exp
            sigma = p['sigma_1']
        else:
            exp_func = exp
            sigma = 0
        fun = np.zeros(len(x))
        comps = {}
        for key in self.sub_functions.keys():
            if key.startswith('exp'):
                for i in range(self.sub_functions[key]['degree']):
                    f = exp_func_normal(str(i + 1), sub_func=key)
                    fun = fun + f
                    comps['exp_' + str(i + 1)] = f
            elif key == 'cos':
                for i in range(self.sub_functions[key]['degree']):
                    f = p['cos_amp_' + str(i + 1)] * np.cos(
                        x * p['cos_omega_' + str(i + 1)]
                        + p['cos_phi_' + str(i + 1)])
                    fun = fun + f
                    comps['cos_' + str(i + 1)] = f
        if 'osc' in self.sub_functions.keys():
            osc_fun = np.zeros(len(x))
            for i in range(self.sub_functions['osc']['degree']):
                f = np.cos(x * p['osc_omega_' + str(i + 1)] +
                           p['osc_phi_' + str(i + 1)])
                osc_fun = osc_fun + f
                comps['osc_' + str(i+1)] = f
            for i in range(self.sub_functions['damp']['degree']):
                f = exp_func_normal(str(i + 1), sub_func='damp')
                osc_fun = osc_fun * f
                comps['damp_' + str(i+1)] = f
            fun = fun * osc_fun
        if 'const' in self.sub_functions.keys():
            f = p['const_amp_1'] * \
                exp_func(x, np.inf, sigma, t0=p['x0_1'], **kwargs)
            fun = fun + f
            comps['const'] = f
        if return_comps:
            return fun, comps
        else:
            return fun

    def _exp_simple(self, params, *args, **kwargs):
        if 'exp_amp_1' not in params.keys():
            params['exp_amp_1'] = 1
        if 'x0_1' not in params.keys():
            params['x0_1'] = 0
        return lambda x: params['exp_amp_1']*exp(x, params['tau_1'],
                                                 t0=params['x0_1'], **kwargs)

    def _hermite_poly_function(self, x, p, *args, return_comps=False,
                               **kwargs):
        def hermite_func(n, sig, x0, x):
            def hermite_poly_recur(i):
                if i == 0:
                    return np.array([1])
                elif i == 1:
                    return np.array([2, 0])
                else:
                    h1 = np.zeros((1, i + 1))[0]
                    h1[0:i] = 2*hermite_poly_recur(i - 1)
                    h2 = np.zeros((1, i + 1))[0]
                    h2[2:] = 2*(i - 1)*hermite_poly_recur(i - 2)
                    return h1 - h2

            gauss = np.exp(-((x-x0)/sig)**2)
            h = hermite_poly_recur(n)
            return (1/np.sqrt((2**n)*np.math.factorial(n)) *
                    np.polyval(h, (x-x0)/sig)*gauss)
        fun = np.zeros(len(x))
        comps = {}
        for key in self.sub_functions.keys():
            if re.search('hermit', key, re.I):
                for name in (
                        self.sub_functions[key]['para_names']['individual']):
                    if re.search('amp', name, re.I):
                        ampname = name
                        break
                try:
                    ampname = ampname[:re.search('\d+', ampname).span()[0]]
                except Exception:
                    pass
                for i in range(self.sub_functions[key]['degree']):
                    f = p["_".join([ampname, str(i + 1)])] * \
                        hermite_func(i, p['sigma_1'], p['x0_1'], x)
                    fun = fun + f
                    comps['hermite_' + str(i + 1)] = f
            elif re.search('const', key, re.I):
                fun += p['const_amp_1']
                comps['const'] = p['const_amp_1']
        if return_comps:
            return fun, comps
        else:
            return fun

    def _polynomial_function(self, params, *args, **kwargs):
        p = [params[para].value for para in params]
        return np.poly1d(p)

    def _fit_function_wrap(self, params, func, *args, **kwargs):
        return lambda x: func(x, params, *args, **kwargs)

    def calc_residual(self, params, queue=None, **kwargs):
        fun = self.fit_function(params, **kwargs)
        if self.dy is None:
            residual = self.y - fun(self.x)
        else:
            residual = ((self.y) - fun(self.x))/self.dy
        return residual


def lmfitreport_to_dict(report):
    sections = re.split("\[\[|\]\]", report)
    content = {}
    dct = {"type": "label", "grid_kwargs": {"sticky": "wn"}}
    if len(sections) <= 1:
        dct["content"] = {"": {"text": report, "visible": True}}
    else:
        i = 1
        while i < len(sections) - 1:
            content[sections[i]] = {"text": sections[i + 1], "visible": True}
            i += 2
        try:
            content['Correlations']["visible"] = False
        except Exception:
            pass
        dct["content"] = content
    return dct
