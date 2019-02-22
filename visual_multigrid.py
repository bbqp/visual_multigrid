import math
import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import numpy.polynomial.legendre as npleg

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

"""
visual_multigrid.py

Code that visually runs through a multigrid V-cycle.
"""

class VisualGrid:
    def __init__(self, x0 = 0, xn = 1, y0 = 0, yn = 1, max_levels = 5, rhsf = None, imag_dir = 'images', err_dir = 'errors'):
        self._x0 = x0
        self._xn = xn
        self._y0 = y0
        self._yn = yn

        self._numx0 = 3
        self._numy0 = 3

        self._current_level = max_levels - 1
        self._max_levels = max_levels

        self._rhsf = rhsf
        self._image_dir = imag_dir
        self._err_dir = err_dir

        self._numx = 2**self._current_level * (self._numx0 - 1) + 1
        self._numy = 2**self._current_level * (self._numy0 - 1) + 1

        self._hx = (self._xn - self._x0) / (self._numx - 1)
        self._hy = (self._yn - self._y0) / (self._numy - 1)

        self._image_counter = 0

    def refine(self):
        return self.jump_to_level(self._current_level + 1)

    def coarsen(self):
        return self.jump_to_level(self._current_level - 1)

    def jump_to_level(self, k):
        if 0 <= k and k < self._max_levels:
            self._current_level = k

            self._numx = 2**k * (self._numx0 - 1) + 1
            self._numy = 2**k * (self._numy0 - 1) + 1

            self._hx = (self._xn - self._x0) / (self._numx - 1)
            self._hy = (self._yn - self._y0) / (self._numy - 1)

        return self

    def current_level(self):
        return self._current_level

    def max_levels(self):
        return self._max_levels

    def xbounds(self):
        return (self._x0, self._xn)

    def ybounds(self):
        return (self._y0, self._yn)

    def xy_point_counts(self):
        return (self._numx, self._numy)

    def xy_steps(self):
        return (self._hx, self._hy)

    def error_directory(self):
        return self._err_dir

    def image_directory(self):
        return self._image_dir

    def run(self, niters = 10, resid_tol = 1e-3, gamma = 1, exact = None, zlim = None, saveplot = False, showplot = False):
        # Set the initial guess on the grid.
        U = np.zeros(self.xy_point_counts())
        U[1:(self._numx - 1), 1:(self._numy - 1)] = np.reshape(-1.0 + 2.0 * npr.rand(self._numx - 2, self._numy - 2), (self._numx - 2, self._numy - 2))

        # Set the right-hand-side matrix.
        F = compute_rhs(self)

        # Create a grid of points on which we can plot grid functions.
        X, Y = np.meshgrid(np.linspace(self._x0, self._xn, self._numx),
                            np.linspace(self._y0, self._yn, self._numy))
        X = X.T
        Y = Y.T

        # Define an array for successive approximations.
        V = np.zeros(U.shape)

        # Create a vectorized version of the exact solution.
        if exact is not None:
            exact_vec = np.vectorize(exact)

        # Create the initial residual and store its norm, along with a history
        # of residual norms in the iterations to follow.
        R = compute_residual(self, U, F)
        residual_norm_history = [self.l2_norm(R)]

        # On the off-chance we guess the correct solution, we bounce.
        if residual_norm_history[-1] < resid_tol:
            lucky_zmax = max(np.max(abs(U)), 1)
            zlim = [-lucky_zmax, lucky_zmax]
            plot_gridfunction(grid, U, 'Lucky Initial Guess',
                directory=self._image_dir, filename='luckyguess.png',
                zlim=zlim, saveplot=saveplot, showplot=showplot)

            print('You provided a close-enough guess with a residual norm of {1:.2e}'.format(residual_norm_history[-1]))

            return

        iterations = 0

        while iterations < niters:
            print('Iteration {0:d} of {1:d}, current residual norm = {2:.2e}'.format(iterations, niters - 1, residual_norm_history[-1]))

            # Compute the numerical solution.
            V[:] = visual_gamma_cycle(self, U, F, gamma = gamma, saveplot = saveplot, showplot = showplot)

            if exact is not None:
                # Print the numerical error to file.
                absolute_error = np.absolute(exact_vec(X, Y) - V)
                l2_err = self.l2_error(exact, V)

                plot_title = 'Absolute Error on Iteration ' + str(iterations) + \
                                '\n' + \
                                '$L^2$ Error Norm = {0:.2e}'.format(l2_err)

                maxz = max(np.max(np.abs(absolute_error)), 1.0)
                abserr_zlim = [-maxz, maxz]
                plot_gridfunction(grid, absolute_error, plot_title,
                        directory = self._err_dir, filename = 'abserror_{0:04d}.png'.format(iterations), zlim = abserr_zlim, saveplot = saveplot, showplot = showplot)

            # Compute the new residual and add its norm to our history.
            R = compute_residual(self, V, F)
            residual_norm_history.append(self.l2_norm(R))

            plot_title = 'Residual on Iteration ' + str(iterations) + \
                            '\n' + \
                            '$L^2$ Norm = {0:.2e}'.format(residual_norm_history[-1])

            # Plot what we will normally have access to: The residuals.
            maxz = max(np.max(np.abs(R)), 1.0)
            residual_zlim = [-maxz, maxz]
            plot_gridfunction(grid, R, plot_title,
                    directory = self._err_dir, filename = 'residual_{0:04d}.png'.format(iterations), zlim = residual_zlim, saveplot = saveplot, showplot = showplot)

            # If we fall below the specified residual norm, break.
            if (residual_norm_history[-1] / residual_norm_history[0]) < resid_tol:
                break

            # Set the previous guess to be the solution we just computed.
            U[:] = V

            # Increment the iteration count.
            iterations += 1

        if iterations < niters:
            print('Multigrid converged in {0:d} iterations with absolute residual error {1:.2e}, relative residual error {2:.2e}'.format(iterations + 1, residual_norm_history[-1], residual_norm_history[-1] / residual_norm_history[0]))
        else:
            print('Multigrid may not have converged: {0:d} iterations were reached with absolute residual error {1:.2e}, relative residual error {2:.2e}'.format(iterations, residual_norm_history[-1], residual_norm_history[-1] / residual_norm_history[0]))

        # Plot and save the residual norm history over the course of the iterations.
        if iterations > 0:
            resid_fig = plt.figure()
            plt.loglog(1 + np.linspace(0, iterations + 1, iterations + 2), residual_norm_history, 'r^--', linewidth = 0.5)
            plt.xlabel('Number of Guesses')
            plt.ylabel('Residual $L^2$ Norm')
            plt.title('Residual Norm History for Multigrid')

            if showplot:
                plt.show()

            if saveplot:
                plt.savefig('errors\\\\residual_history.png')

            # Now compute the error ratios and apparent convergence rates.
            residual_norm_history = np.array(residual_norm_history)
            ratios = residual_norm_history[:-1] / residual_norm_history[1:]
            rates = np.log(ratios) / np.log(10)

            rates_fig = plt.figure()
            plt.semilogy(np.linspace(0, iterations, iterations + 1), ratios, 'b^--', linewidth = 0.5)
            plt.title('Successive Numerical Residual Ratios')

            if showplot:
                plt.show()

            if saveplot:
                plt.savefig('errors\\\\resid_ratios.png')

    def l2_error(self, exact_func, U):
        l2_err_sq = 0

        for j in range(self._numy - 1):
            y = self._y0 + j * self._hy / 2

            for i in range(self._numx - 1):
                x = self._x0 + i * self._hx / 2

                l2_err_sq += ( exact_func(x, y) - 0.25 * np.sum(U[i:(i + 2), j:(j + 2)]) )**2

        return math.sqrt( l2_err_sq * (self._hx * self._hy) )
    
    def l2_error_gauss(self, exact_func, U, n = 1):
        # Keep track of the square of the L2 error.
        l2_err_sq = 0

        # Placeholders for the interval endpoints and grid step sizes.
        x0 = self._x0
        y0 = self._y0
        hx = self._hx
        hy = self._hy
        
        # Compute the gauss points and weights on the interval [-1,1],
        # which give an exact quadrature for polynomials of degree up to
        # 2n - 1.
        p, w = npleg.leggauss(n)

        # Loop through the cells of the grid in lexicographic order, starting
        # from the lower left corner.
        for j in range(self._numy - 1):
            y = y0 + j * hy
            
            # Translate the quadrature points to the current y-interval.
            py = y + 0.5 * hy * (p + 1)

            for i in range(self._numx - 1):
                x = x0 + i * hx
                
                # Translate the quadrature points to the current x-interval.
                px = x + 0.5 * hx * (p + 1)
                
                # Create the bilinear function on this cell.
                blfunc = bilinear_func(U[i:i+1, j:j+1], hx, hy)
                
                # Compute the approximate integral (sans the jacobian determinant)
                # of the difference between the exact and approximate functions.
                for m in range(len(px)):
                    for n in range(len(py)):                        
                        l2_err_sq += w[m] * w[n] * (exact_func(px[m], py[n]) - blfunc(px[m], py[n]))**2

        return math.sqrt(l2_err_sq * (0.25 * hx * hy))

    def l2_norm(self, U, n = 1):
        # Keep track of the square of the L2 error.
        l2_norm_sq = 0
        
        # Placeholders for the interval endpoints and grid step sizes.
        x0 = self._x0
        y0 = self._y0
        hx = self._hx
        hy = self._hy
        
        # Compute the gauss points and weights on the interval [-1,1],
        # which give an exact quadrature for polynomials of degree up to
        # 2n - 1.
        p, w = npleg.leggauss(n)

        # Loop through the cells of the grid in lexicographic order, starting
        # from the lower left corner.
        for j in range(self._numy - 1):
            y = y0 + j * hy
            
            # Translate the quadrature points to the current y-interval.
            py = y + 0.5 * hy * (p + 1)

            for i in range(self._numx - 1):
                x = x0 + i * hx
                
                # Translate the quadrature points to the current x-interval.
                px = x + 0.5 * hx * (p + 1)
                
                # Create the bilinear function on this cell.
                blfunc = bilinear_func(U[i:i+2, j:j+2], hx, hy, x, y)
                
                # Compute the approximate integral (sans the jacobian determinant)
                # of the grid function on the cell.
                for m in range(len(px)):
                    for n in range(len(py)):                        
                        l2_norm_sq += w[m] * w[n] * blfunc(px[m], py[n])**2

        return math.sqrt(l2_norm_sq * (0.25 * hx * hy))


def bilinear_func(U, hx, hy, x1, y1):
    # Compute the coefficients of the bilinear functions.
    c = np.zeros(4)
    
    c[0] = U[0, 0]
    c[1] = (U[1, 0] - U[0, 0]) / hx
    c[2] = (U[0, 1] - U[0, 0]) / hy
    c[3] = (U[1,1] - U[1,0] - U[0,1] + U[0,0]) / (hx * hy)
    
    return lambda x, y : c[0] + \
                         c[1] * (x - x1) + \
                         c[2] * (y - y1) + \
                         c[3] * (x - x1) * (y - y1)


def visual_vcycle(grid, U, F, saveplot = False, showplot = False):
    #---------------------------------------------#
    # Plot the initial guess on the current grid. #
    #---------------------------------------------#

    if grid.current_level() == grid.max_levels() - 1:
        plot_title = 'Initial Guess on Grid ' + str(grid._current_level)
    else:
        plot_title = 'Initial Correction on Grid ' + str(grid._current_level)

    plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

    if grid.current_level() > 0:
        #--------------------------------------------------------------#
        # Perform one sweep of Gauss Seidel and plot the next iterate. #
        #--------------------------------------------------------------#

        if grid._current_level == grid._max_levels - 1:
            plot_title = 'Pre-Sweep Solution on Grid ' + str(grid._current_level)
        else:
            plot_title = 'Pre-Sweep Correction on Grid ' + str(grid._current_level)

        U[:] = gauss_seidel(grid, U, F, plot_title, saveplot = saveplot, showplot = showplot)

        # Now compute the residual and restrict the residual to a coarser grid.
        R = compute_residual(grid, U, F)
        RC = restrict(R)

        grid.coarsen()

        # Define the coarse error as the zero function for the initial guess
        # and solve the same problem on a coarser grid.
        EC = np.zeros(RC.shape)
        EC[:] = visual_vcycle(grid, EC, RC, saveplot = saveplot, showplot = showplot)

        #------------------------------------------------------#
        # Plot the corrected solution prior to the post-sweep. #
        #------------------------------------------------------#

        grid.refine()

        # Interpolate the coarse error and add it to the solution.
        EF = interp(EC)
        U += EF

        if grid.current_level() == grid.max_levels() - 1:
            plot_title = 'Addition of Error to Solution on Grid ' + str(grid._current_level)
        else:
            plot_title = 'Addition of Error to Correction on Grid ' + str(grid._current_level)

        plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

        #-------------------------------------------------------#
        # Plot the corrected solution following the post-sweep. #
        #-------------------------------------------------------#

        if grid.current_level() == grid.max_levels() - 1:
            plot_title = 'Post-Sweep of Solution on Grid ' + str(grid._current_level)
        else:
            plot_title = 'Post-Sweep of Error to Correction on Grid ' + str(grid._current_level)

        # Perform one post sweep on the numerical solution.
        U[:] = gauss_seidel(grid, U, F, plot_title, saveplot = saveplot, showplot = showplot)
    else:
        #--------------------------------------------------#
        # Compute the exact solution on the coarsest grid. #
        #--------------------------------------------------#

        U[:] = exact_solve(grid, U, F)

        plot_title = 'Exact Correction on Grid ' + str(grid._current_level)
        plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

    return U


def visual_gamma_cycle(grid, U, F, gamma = 1, saveplot = False, showplot = False):
    #---------------------------------------------#
    # Plot the initial guess on the current grid. #
    #---------------------------------------------#

    if grid.current_level() == grid.max_levels() - 1:
        plot_title = 'Initial Guess on Grid ' + str(grid._current_level)
    else:
        plot_title = 'Initial Correction on Grid ' + str(grid._current_level)

    plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

    if grid.current_level() > 0:
        #--------------------------------------------------------------#
        # Perform one sweep of Gauss Seidel and plot the next iterate. #
        #--------------------------------------------------------------#

        if grid._current_level == grid._max_levels - 1:
            plot_title = 'Pre-Sweep Solution on Grid ' + str(grid._current_level)
        else:
            plot_title = 'Pre-Sweep Correction on Grid ' + str(grid._current_level)

        U[:] = gauss_seidel(grid, U, F, plot_title, saveplot = saveplot, showplot = showplot)

        # Now compute the residual and restrict the residual to a coarser grid.
        R = compute_residual(grid, U, F)
        RC = restrict(R)

        grid.coarsen()

        # Define the coarse error as the zero function for the initial guess
        # and solve the same problem on a coarser grid.
        EC = np.zeros(RC.shape)

        for k in range(gamma):
            EC[:] = visual_gamma_cycle(grid, EC, RC, gamma = gamma, saveplot = saveplot, showplot = showplot)

        #------------------------------------------------------#
        # Plot the corrected solution prior to the post-sweep. #
        #------------------------------------------------------#

        grid.refine()

        # Interpolate the coarse error and add it to the solution.
        EF = interp(EC)
        U += EF

        if grid.current_level() == grid.max_levels() - 1:
            plot_title = 'Addition of Error to Solution on Grid ' + str(grid._current_level)
        else:
            plot_title = 'Addition of Error to Correction on Grid ' + str(grid._current_level)

        plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

        #-------------------------------------------------------#
        # Plot the corrected solution following the post-sweep. #
        #-------------------------------------------------------#

        if grid.current_level() == grid.max_levels() - 1:
            plot_title = 'Post-Sweep of Solution on Grid ' + str(grid._current_level)
        else:
            plot_title = 'Post-Sweep of Error to Correction on Grid ' + str(grid._current_level)

        # Perform one post sweep on the numerical solution.
        U[:] = gauss_seidel(grid, U, F, plot_title, saveplot = saveplot, showplot = showplot)
    else:
        #--------------------------------------------------#
        # Compute the exact solution on the coarsest grid. #
        #--------------------------------------------------#

        U[:] = exact_solve(grid, U, F)

        plot_title = 'Exact Correction on Grid ' + str(grid._current_level)
        plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

    return U


def plot_gridfunction(grid, U, plot_title, directory = None, filename = None, zlim = [-1, 1], line = None, saveplot = False, showplot = False):
    #---------------------------------------#
    # Create the meshgrid on which to plot. #
    #---------------------------------------#

    X, Y = np.meshgrid(np.linspace(grid._x0, grid._xn, grid._numx),
                                    np.linspace(grid._y0, grid._yn, grid._numy))

    # Create a figure.
    fig = plt.figure()

    # Set up the 3D plot axes.
    axes = fig.gca(projection = '3d')

    # Create a surface plot of the approximate solution.
    surf = axes.plot_surface(X, Y, U.T, rstride = 1,
                                cstride = 1,
                                cmap = cm.rainbow,
                                linewidth = 0.5,
                                edgecolor = 'black',
                                antialiased = True)

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

    axes.set_xlim([grid._x0, grid._xn])
    axes.set_ylim([grid._y0, grid._yn])

    if zlim != None:
        axes.set_zlim(zlim)

    if line is not None:
        axes.plot3D(X[line, :], Y[line, :], U[:, line], linewidth=2.0,
            color='hotpink')

    plt.title(plot_title)

    if saveplot:
        if filename is None:
            image_filename = 'mgimage{0:04d}.png'.format(grid._image_counter)
        else:
            image_filename = filename

        if directory is not None:
            image_filename = directory + '\\\\' + image_filename

        plt.savefig(image_filename)

    if showplot:
        plt.show()
    else:
        plt.close(fig)

    grid._image_counter += 1


def compute_rhs(grid):
    numx, numy = grid.xy_point_counts()
    hx, hy = grid.xy_steps()
    x0, xn = grid.xbounds()
    y0, yn = grid.ybounds()

    F = np.zeros((numx, numy))
    rhsf = grid._rhsf

    if rhsf is not None:
        for j in range(numy):
            for i in range(numx):
                F[i, j] = rhsf(x0 + i * hx, y0 + j * hy)

    return F


def compute_residual(grid, U, F):
    numx, numy = grid.xy_point_counts()
    hx, hy = grid.xy_steps()

    R = np.zeros((numx, numy))

    for j in range(1, numy - 1):
        for i in range(1, numx - 1):
            R[i, j] = F[i, j] + ( U[i - 1, j] - 2 * U[i, j] + U[i + 1, j] ) / hx**2 + \
                                ( U[i, j - 1] - 2 * U[i, j] + U[i, j + 1] ) / hy**2

    return R


def gauss_seidel(grid, U, F, plot_title, saveplot = False, showplot = False):
    numx, numy = grid.xy_point_counts()
    hx, hy = grid.xy_steps()

    alpha = (hx / hy)**2

    plotted = False

    for j in range(1, numy - 1):
        for i in range(1, numx - 1):
            U[i, j] = (
                hx**2 * F[i, j] + \

                U[i - 1, j] + U[i + 1, j] + \

                alpha * ( U[i, j - 1] + U[i, j + 1] )
            ) / (2 * (1 + alpha))

        if grid.current_level() == grid.max_levels() - 1:
            if not plotted:
                plotted = True

            plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), line = j, saveplot = saveplot, showplot = showplot)

    if not plotted:
        plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

    return U


def wsor(grid, U, F, w, plot_title, saveplot = False, showplot = False):
    numx, numy = grid.xy_point_counts()
    hx, hy = grid.xy_steps()
    alpha = (hx / hy)**2
    plotted = False

    for j in range(1, numy - 1):
        for i in range(1, numx - 1):
            temp = (
                hx**2 * F[i, j] + \

                U[i - 1, j] + U[i + 1, j] + \

                alpha * ( U[i, j - 1] + U[i, j + 1] )
            ) / (2 * (1 + alpha))

            U[i, j] = w * temp + (1 - w) * U[i, j]

        if grid.current_level() == grid.max_levels() - 1:
            if not plotted:
                plotted = True

            plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

    if not plotted:
        plot_gridfunction(grid, U, plot_title, directory = grid.image_directory(), saveplot = saveplot, showplot = showplot)

    return U


def restrict(UF):
    MC = UF.shape[0] // 2 + 1
    NC = UF.shape[1] // 2 + 1

    UC = np.zeros((MC, NC))

    for j in range(1, NC - 1):
        for i in range(1, MC - 1):
            UC[i, j] = (
                4.0 * UF[2*i, 2*j] +

                2.0 * (
                    UF[2*i - 1, 2*j    ] +
                    UF[2*i + 1, 2*j    ] +
                    UF[2*i    , 2*j - 1] +
                    UF[2*i    , 2*j + 1]
                ) +

                UF[2*i - 1, 2*j - 1] +
                UF[2*i + 1, 2*j - 1] +
                UF[2*i - 1, 2*j + 1] +
                UF[2*i + 1, 2*j + 1]
            ) / 16.0

    return UC


def interp(UC):
    M = 2 * UC.shape[0] - 1
    N = 2 * UC.shape[1] - 1
    UF = np.zeros((M, N))

    UF[0:M:2, 0:N:2] = UC

    # Interpolate the coarse x-lines.
    for i in range(0, M, 2):
        UF[i, 1:(N - 1):2] = 0.5 * (UF[i, 0:(N - 2):2] + UF[i, 2:N:2])

    # Interpolate the coarse y-lines.
    for j in range(0, N, 2):
        UF[1:(M - 1):2, j] = 0.5 * (UF[0:(M - 2):2, j] + UF[2:M:2, j])

    # Interpolate the cell-centered points.
    for j in range(1, N - 1, 2):
        for i in range(1, M - 1, 2):
            UF[i, j] = 0.25 * (UF[i - 1, j - 1] + UF[i + 1, j - 1] + UF[i - 1, j + 1] + UF[i + 1, j + 1])

    return UF


def exact_solve(grid, U, F):
    U[1, 1] = grid._hx**2 * F[1, 1]

    return U


if __name__ == '__main__':
    def zerofunction(x, y): return 0.0
    def demorhs(x, y): return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    def demoexact(x, y): return np.sin(np.pi * x) * np.sin(np.pi * y)

    niters = 10
    max_levels = 5
    resid_tol = 1e-2
    gamma = 1

    grid = VisualGrid(max_levels = max_levels, rhsf = demorhs)
    grid.run(niters = niters, resid_tol = resid_tol, gamma = gamma, exact = demoexact, zlim = [-1.0, 1.0], saveplot = True)
