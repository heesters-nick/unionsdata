import logging
from typing import Literal

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, label

from unionsdata.config import BandCfg

logger = logging.getLogger(__name__)


def preprocess_cutout(
    cutout: NDArray[np.float32],
    bands: list[str],
    in_dict: dict[str, BandCfg],
    standard_zp: float = 30.0,
) -> NDArray[np.float32]:
    """Create an RGB image from the cutout data and save or plot it.

    Args:
        cutout: cutout data with shape (channels, height, width)
        bands: list of band names corresponding to cutout channels
        in_dict: dictionary with band-specific configuration
        standard_zp: standard zero-point to adjust fluxes

    Returns:
        preprocessed image cutout

    Raises:
        ValueError: If number of bands is not exactly 3.

    """
    if cutout.shape[0] != 3:
        raise ValueError(f'Expected cutout with 3 bands, got {cutout.shape[0]}')

    if len(bands) != 3:
        raise ValueError(f'Expected 3 band names, got {len(bands)}')

    # Get band order from dictionary (shortest to longest wavelength)
    all_bands_ordered = list(in_dict.keys())

    # Sort the input bands by their position in the dictionary
    # Format: [(band_name, dict_position, original_index), ...]
    band_info = [(band, all_bands_ordered.index(band), i) for i, band in enumerate(bands)]
    band_info_sorted = sorted(band_info, key=lambda x: x[1])

    # Shortest wavelength → blue, middle → green, longest → red
    blue_band, _, blue_idx = band_info_sorted[0]
    green_band, _, green_idx = band_info_sorted[1]
    red_band, _, red_idx = band_info_sorted[2]

    logger.debug(f'RGB mapping: R={red_band}, G={green_band}, B={blue_band}')

    # Extract channels from cutout
    cutout_red = cutout[red_idx].copy()
    cutout_green = cutout[green_idx].copy()
    cutout_blue = cutout[blue_idx].copy()

    # Apply flux adjustments to standard zero-point
    if np.count_nonzero(cutout_red) > 0:
        current_zp = in_dict[red_band].zp
        if current_zp != standard_zp:
            cutout_red = adjust_flux_with_zp(cutout_red, current_zp, standard_zp)

    if np.count_nonzero(cutout_green) > 0:
        current_zp = in_dict[green_band].zp
        if current_zp != standard_zp:
            cutout_green = adjust_flux_with_zp(cutout_green, current_zp, standard_zp)

    if np.count_nonzero(cutout_blue) > 0:
        current_zp = in_dict[blue_band].zp
        if current_zp != standard_zp:
            cutout_blue = adjust_flux_with_zp(cutout_blue, current_zp, standard_zp)

    # replace anomalies
    cutout_red = detect_anomaly(cutout_red)
    cutout_green = detect_anomaly(cutout_green)
    cutout_blue = detect_anomaly(cutout_blue)

    # synthesize missing channel from the existing ones
    # longest valid wavelength is mapped to red, middle to green, shortest to blue
    if np.count_nonzero(cutout_red > 1e-10) == 0:
        cutout_red = cutout_green
        cutout_green = (cutout_green + cutout_blue) / 2
    elif np.count_nonzero(cutout_green > 1e-10) == 0:
        cutout_green = (cutout_red + cutout_blue) / 2
    elif np.count_nonzero(cutout_blue > 1e-10) == 0:
        cutout_blue = cutout_green
        cutout_green = (cutout_red + cutout_blue) / 2

    # stack the channels in the order red, green, blue
    cutout_prep = np.stack([cutout_red, cutout_green, cutout_blue], axis=-1).astype(np.float32)

    return cutout_prep


def generate_rgb(
    cutout: NDArray[np.float32],
    scaling_type: Literal['asinh', 'linear'] = 'asinh',
    stretch: float = 125,
    Q: float = 7.0,
    gamma: float = 0.25,
) -> NDArray[np.float32]:
    """Create an RGB image from three bands of data preserving relative intensities.

    Processes multi-band astronomical data into a properly scaled RGB image
    suitable for visualization, handling high dynamic range and empty channels.

    Args:
        cutout: 3D array of shape (height, width, 3) with band data
        scaling_type: Type of scaling to apply ("asinh" or "linear")
        stretch: Scaling factor controlling overall brightness
        Q: Softening parameter for asinh scaling (higher = more linear)
        gamma: Gamma correction factor (lower = enhances faint features)

    Returns:
        Normalized RGB image with values in range [0, 1]

    Notes:
        For astronomical data with high dynamic range, "asinh" scaling is
        typically preferred as it preserves both bright and faint details.
    """
    frac = 0.1
    with np.errstate(divide='ignore', invalid='ignore'):
        red = cutout[:, :, 0]
        green = cutout[:, :, 1]
        blue = cutout[:, :, 2]

        # Check for zero channels
        red_is_zero = np.all(red == 0)
        green_is_zero = np.all(green == 0)
        blue_is_zero = np.all(blue == 0)

        # Compute average intensity before scaling choice (avoiding zero channels)
        nonzero_channels = []
        if not red_is_zero:
            nonzero_channels.append(red)
        if not green_is_zero:
            nonzero_channels.append(green)
        if not blue_is_zero:
            nonzero_channels.append(blue)

        if nonzero_channels:
            i_mean = np.asarray(sum(nonzero_channels) / len(nonzero_channels), dtype=np.float32)
        else:
            i_mean = np.zeros_like(red, dtype=np.float32)  # All channels are zero

        if scaling_type == 'asinh':
            # Apply asinh scaling
            if not red_is_zero:
                red = (
                    red * np.arcsinh(Q * i_mean / stretch) * frac / (np.arcsinh(frac * Q) * i_mean)
                )
            if not green_is_zero:
                green = (
                    green
                    * np.arcsinh(Q * i_mean / stretch)
                    * frac
                    / (np.arcsinh(frac * Q) * i_mean)
                )
            if not blue_is_zero:
                blue = (
                    blue * np.arcsinh(Q * i_mean / stretch) * frac / (np.arcsinh(frac * Q) * i_mean)
                )
        elif scaling_type == 'linear':
            # Apply linear scaling
            if not red_is_zero:
                red = red * stretch
            if not green_is_zero:
                green = green * stretch
            if not blue_is_zero:
                blue = blue * stretch
        else:
            raise ValueError(f'Unknown scaling type: {scaling_type}')

        # Apply gamma correction while preserving sign
        if gamma is not None:
            if not red_is_zero:
                red_mask = abs(red) <= 1e-9
                red = np.sign(red) * (abs(red) ** gamma)
                red[red_mask] = 0

            if not green_is_zero:
                green_mask = abs(green) <= 1e-9
                green = np.sign(green) * (abs(green) ** gamma)
                green[green_mask] = 0

            if not blue_is_zero:
                blue_mask = abs(blue) <= 1e-9
                blue = np.sign(blue) * (abs(blue) ** gamma)
                blue[blue_mask] = 0
        # Stack the channels after scaling and gamma correction
        result = np.stack([red, green, blue], axis=-1).astype(np.float32)
        # Clip the result to [0, 1] for display
        result = np.clip(result, 0, 1).astype(np.float32)
    return result


def normalize_mono(
    cutout: NDArray[np.float32],
    scaling_type: Literal['asinh', 'linear'] = 'asinh',
    stretch: float = 125,
    Q: float = 7.0,
    gamma: float = 0.25,
) -> NDArray[np.float32]:
    """Normalize a single-band cutout for display.

    Applies anomaly detection, scaling, and gamma correction to a monochromatic image.

    Args:
        cutout: 2D array of shape (height, width)
        scaling_type: Type of scaling ('asinh' or 'linear')
        stretch: Scaling factor controlling overall brightness
        Q: Softening parameter for asinh scaling
        gamma: Gamma correction factor

    Returns:
        Normalized grayscale image with shape (height, width) and values in [0, 1]
    """
    # Apply anomaly detection
    image = detect_anomaly(cutout.copy())

    # Handle empty images
    if np.count_nonzero(image) == 0:
        return np.zeros(image.shape, dtype=np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        if scaling_type == 'asinh':
            frac = 0.1
            scaled = image * np.arcsinh(Q * image / stretch) * frac / (np.arcsinh(frac * Q) * image)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        elif scaling_type == 'linear':
            scaled = image * stretch
        else:
            raise ValueError(f'Unknown scaling type: {scaling_type}')

        # Apply gamma correction
        if gamma is not None:
            mask = np.abs(scaled) <= 1e-9
            scaled = np.sign(scaled) * (np.abs(scaled) ** gamma)
            scaled[mask] = 0

        # Clip to [0, 1]
        scaled = np.clip(scaled, 0, 1).astype(np.float32)

    return scaled


def adjust_flux_with_zp(
    flux: NDArray[np.float32], current_zp: float | int, standard_zp: float | int
) -> NDArray[np.float32]:
    """
    Adjust flux values to a standard zero-point.

    Args:
        flux: Flux values to adjust.
        current_zp: Current zero-point of the flux values.
        standard_zp: Standard zero-point to adjust to.

    Returns:
        NDArray[np.float32]: Adjusted flux values.
    """
    adjusted_flux = flux * 10 ** (-0.4 * (current_zp - standard_zp))
    return adjusted_flux.astype(np.float32)


def detect_anomaly(
    image: NDArray[np.float32],
    zero_threshold: float = 0.005,
    min_size: int = 50,
    replace_anomaly: bool = True,
    dilate_mask: bool = True,
    dilation_iters: int = 1,
) -> NDArray[np.float32]:
    """
    Detect and replace anomalies in an image using wavelet decomposition.

    This function analyzes an astronomical image to identify anomalous regions
    by performing wavelet decomposition and identifying regions with minimal
    fluctuations below a threshold. It can optionally replace detected anomalous
    pixels with zeros.

    Args:
        image: Input astronomical image to process
        zero_threshold: Fluctuation threshold below which an anomaly is detected
        min_size: Minimum connected pixel count to be considered an anomaly
        replace_anomaly: Whether to set anomalous pixels to zero
        dilate_mask: Whether to expand the detected anomaly mask
        dilation_iters: Number of dilation iterations if dilate_mask is True

    Returns:
        Processed image with anomalies optionally replaced

    Notes:
        This function uses Haar wavelet decomposition to identify regions with
        suspiciously low variation, which often indicate detector artifacts or
        other non-astronomical features in the image.
    """
    # replace nan values with zeros
    image[np.isnan(image)] = 0.0

    # Perform a 2D Discrete Wavelet Transform using Haar wavelets
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs  # Decomposition into approximation and details

    # Create binary masks where wavelet coefficients are below the threshold
    mask_horizontal = np.abs(cH) <= zero_threshold
    mask_vertical = np.abs(cV) <= zero_threshold
    mask_diagonal = np.abs(cD) <= zero_threshold

    masks = [mask_diagonal, mask_horizontal, mask_vertical]

    # Create a global mask to accumulate all anomalies
    global_mask = np.zeros_like(image, dtype=bool)
    # Create masks for each component
    component_masks = np.zeros((3, cA.shape[0], cA.shape[1]), dtype=bool)
    anomalies = np.zeros(3, dtype=bool)
    for i, mask in enumerate(masks):
        # Apply connected-component labeling to find connected regions in the mask
        labeled_array, num_features = label(mask)  # type: ignore

        # Calculate the sizes of all components
        component_sizes = np.bincount(labeled_array.ravel())

        # Check if any component is larger than the minimum size
        anomaly_detected = np.any(component_sizes[1:] >= min_size)
        anomalies[i] = anomaly_detected

        if not anomaly_detected:
            continue

        # Prepare to accumulate a total mask
        total_feature_mask = np.zeros_like(image, dtype=bool)

        # Loop through all labels to find significant components
        for component_label in range(1, num_features + 1):  # Start from 1 to skip background
            if component_sizes[component_label] >= min_size:
                # Create a binary mask for this component
                component_mask = labeled_array == component_label
                # add component mask to component masks
                component_masks[i] |= component_mask
                # Upscale the mask to match the original image dimensions
                upscaled_mask = np.kron(component_mask, np.ones((2, 2), dtype=bool))
                # Accumulate the upscaled feature mask
                total_feature_mask |= upscaled_mask

        # Accumulate global mask
        global_mask |= total_feature_mask
        # Dilate the masks to catch some odd pixels on the outskirts of the anomaly
        if dilate_mask:
            global_mask = binary_dilation(global_mask, iterations=dilation_iters)
            for j, comp_mask in enumerate(component_masks):
                component_masks[j] = binary_dilation(comp_mask, iterations=dilation_iters)
    # Replace the anomaly with zeros
    if replace_anomaly:
        image[global_mask] = 0.0

    return image.astype(np.float32)
