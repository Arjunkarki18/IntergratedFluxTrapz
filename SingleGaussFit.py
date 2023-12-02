import numpy as np
from mpdaf.obj import Cube, Spectrum
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit
from mpdaf.obj import WaveCoord
import warnings
from scipy.ndimage import gaussian_filter1d

# Define the Gaussian function to fit
def gaussian(x, A, mu, sigma, c):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

# Load MUSE cube
cube_filename = 'Object_4_muse_cutout.fits'
cube = Cube(cube_filename, ext=1)

# Find the index of the H-alpha emission line in the cube's spectral axis
ha_wavelength = 8616.889  # Wavelength of H-alpha in Angstroms
ha_fwhm = 6.45       # FWHM of H-alpha in Angstroms 4.312 
ha_index = np.argmin(np.abs(cube.wave.coord() - ha_wavelength))

# Lists to store results
integrated_fluxes_ha = []
integrated_fluxes_ha_err = []
FWHM = []
FWHM_err = []
gauss_center = []
gauss_center_err = []
signal_to_noise = []

# Define the full wavelength range for plotting
plot_range_min = 8595
plot_range_max = 8640

# Define the wavelength range for fitting
fit_range_min = 8605
fit_range_max = 8630

sigma_smoothing = 0

# Define a list of pixel coordinates (x, y) to be excluded from fitting
excluded_pixels = [(1, 1)]  # Add the pixel coordinates you want to exclude

# Suppress all warnings
warnings.simplefilter("ignore")

# Loop through each pixel and measure integrated flux within the specified range
for y_pixel in range(cube.shape[1]):
    for x_pixel in range(cube.shape[2]):
        # Check if the current pixel is in the excluded list
        if (x_pixel, y_pixel) in excluded_pixels:
            # Skip Gaussian fit for this pixel
            integrated_fluxes_ha.append(np.nan)
            integrated_fluxes_ha_err.append(np.nan)
            FWHM.append(np.nan)
            FWHM_err.append(np.nan)
            gauss_center.append(np.nan)
            gauss_center_err.append(np.nan)
            signal_to_noise.append(np.nan)
            continue  # Move to the next pixel

        # Extract spectrum from the current pixel at the H-alpha position
        spectrum = cube[:, y_pixel, x_pixel]
        
        #smoothed_spectrum = gaussian_filter1d(spectrum.data, sigma_smoothing)

        # Extract the data and wavelength coordinates within the full range for plotting
        plot_mask = (spectrum.wave.coord() >= plot_range_min) & (spectrum.wave.coord() <= plot_range_max)
        data_slice_plot = spectrum.data[plot_mask]
        wave_slice_plot = spectrum.wave.coord()[plot_mask]

        # Create a Spectrum object with the sliced data and wavelength coordinates for plotting
        specline_plot = Spectrum(wave=WaveCoord(cdelt=spectrum.wave.get_step(),
                                                crval=wave_slice_plot[0], cunit=u.angstrom),
                                 data=data_slice_plot)

        # Extract the data and wavelength coordinates within the fitting range
        fit_mask = (spectrum.wave.coord() >= fit_range_min) & (spectrum.wave.coord() <= fit_range_max)
        data_slice_fit = spectrum.data[fit_mask]
        wave_slice_fit = spectrum.wave.coord()[fit_mask]


        # Define initial parameter estimates for the Gaussian fit
        p0 = [np.max(data_slice_fit), ha_wavelength, ha_fwhm / 2.355, np.mean(data_slice_fit)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(gaussian, wave_slice_fit, data_slice_fit, p0=p0, maxfev= 100000)
        # Perform Gaussian fit using curve_fit
        
        popt, pcov = curve_fit(gaussian, wave_slice_fit, data_slice_fit, p0=p0, maxfev= 100000)

        # Extract the Gaussian fit parameters
        A_fit, mu_fit, sigma_fit, c_fit = popt
        err_A_fit, err_mu_fit, err_sigma_fit, err_c_fit = np.sqrt(np.diag(pcov))

        # Calculate S/N
        sn_flux = A_fit / err_A_fit
        sn_fwhm = sigma_fit / err_sigma_fit  # Signal-to-noise for FWHM

        # Only consider fits where S/N >= 1 and sn_fwhm > 3
        if sn_flux >= 3 and sn_fwhm >= 3:
            integrated_fluxes_ha.append(np.abs(A_fit * sigma_fit * np.sqrt(2 * np.pi)))
            integrated_fluxes_ha_err.append(np.abs(err_A_fit * sigma_fit * np.sqrt(2 * np.pi)))
            FWHM.append(sigma_fit * 2.355)
            FWHM_err.append(err_sigma_fit * 2.355)
            gauss_center.append(mu_fit)
            gauss_center_err.append(err_mu_fit)
            # Define a new wave_slice_fit with more points for smoother curve
            new_wave_slice_fit = np.linspace(fit_range_min, fit_range_max, 1000)

            # Calculate the fitted Gaussian curve using the new wave_slice_fit
            fitted_curve = gaussian(new_wave_slice_fit, *popt)

            # Plot Gaussian fit with the smoother curve
            plt.figure(figsize=(8, 6))
            plt.step(wave_slice_plot, data_slice_plot,  color='b', where='mid')
            plt.step(new_wave_slice_fit, fitted_curve,  color='r' , where='mid')  # Use the new_wave_slice_fit
            plt.title(f'Pixel (x, y): ({x_pixel}, {y_pixel})')
            plt.xlabel('Wavelength (Angstroms)')
            plt.ylabel('Flux')
            plt.xlim(plot_range_min, plot_range_max)

            # Format Gaussian fit parameters as a string
            param_str = f"Integrated Flux: {integrated_fluxes_ha[-1]:.2f} ± {integrated_fluxes_ha_err[-1]:.2f}\n" \
                        f"Center: {gauss_center[-1]:.2f} ± {gauss_center_err[-1]:.2f}\n" \
                        f"FWHM: {FWHM[-1]:.2f} ± {FWHM_err[-1]:.2f}\n" \
                        f"S/N (Flux): {sn_flux:.2f}\n" \
                        f"S/N (FWHM): {sn_fwhm:.2f}"

            # Print Gaussian fit parameters on the plot
            plt.text(0.05, 0.85, param_str, transform=plt.gca().transAxes, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.8))

            # Add pixel position information as text annotations
            plt.text(0.05, 0.05, f"Pixel (x, y): ({x_pixel}, {y_pixel})", transform=plt.gca().transAxes,
                     fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.show()

        else:
            # Unable to fit Gaussian, add NaN values
            integrated_fluxes_ha.append(np.nan)
            integrated_fluxes_ha_err.append(np.nan)
            FWHM.append(np.nan)
            FWHM_err.append(np.nan)
            gauss_center.append(np.nan)
            gauss_center_err.append(np.nan)
            signal_to_noise.append(np.nan)

