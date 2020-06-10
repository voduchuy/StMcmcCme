from pypacmensl.ssa.ssa import SSASolver

class full_cme:
    stoich_mat = np.array([ [1],
                            [ -1]
                            ])
    x0 = np.array([[1, 0, 0]])


    def propensity(self, reaction, x, out):
        if reaction is 0:
            out[:] = 1.0
        if reaction is 1:
            out[:] = x[:,0]

    def t_fun(self, t, out):
        out[0] = 1.0
        out[1] = 0.001

class surrogate_cme:
    stoich_mat = np.array([ [1],
                            [ -1]
                            ])
    x0 = np.array([[1]])

    copymax = 100

    def propensity(self, reaction, x, out):
        if reaction is 0:
            out[:] = 1.0*(x[:,0] < copymax)
        if reaction is 1:
            out[:] = x[:,0]*(x[:,0] < copymax)

    def t_fun(self, t, out):
        out[0] = 1.0
        out[1] = 0.001

t_f = 1000