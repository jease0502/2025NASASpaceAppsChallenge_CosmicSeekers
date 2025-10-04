// This file acts as a single source of truth for how to display exoplanet data.
export const parameterMap = {
    // Planet Parameters
    disposition: { label: 'Disposition', group: 'planet' },
    period: { label: 'Orbital Period', unit: 'days', group: 'planet', format: (v) => v.toFixed(2) },
    planet_radius: { label: 'Planet Radius', unit: 'Earth radii', group: 'planet', format: (v) => v.toFixed(2) },
    equilibrium_temp: { label: 'Equilibrium Temperature', unit: 'K', group: 'planet', format: (v) => v ? v.toFixed(0) : 'N/A' },
    insolation_flux: { label: 'Insolation Flux', unit: 'Earth flux', group: 'planet', format: (v) => v ? v.toFixed(2) : 'N/A' },
    duration: { label: 'Transit Duration', unit: 'hours', group: 'planet', format: (v) => v ? v.toFixed(2) : 'N/A' },
    depth: { label: 'Transit Depth', unit: 'ppm', group: 'planet', format: (v) => v ? v.toFixed(0) : 'N/A' },
    snr: { label: 'Signal-to-Noise Ratio (S/N)', group: 'planet', format: (v) => v ? v.toFixed(1) : 'N/A' },
    transit_midpoint: { label: 'Transit Midpoint Time', unit: 'BJD', group: 'planet', format: (v) => v ? v.toFixed(2) : 'N/A' },
    planet_to_star_radius_ratio: { label: 'Planet-to-Star Radius Ratio', group: 'planet', format: (v) => v ? v.toFixed(4) : 'N/A'},

    // Stellar Parameters
    stellar_temp: { label: 'Stellar Temperature', unit: 'K', group: 'star', format: (v) => v ? v.toFixed(0) : 'N/A' },
    stellar_radius: { label: 'Stellar Radius', unit: 'Solar radii', group: 'star', format: (v) => v ? v.toFixed(2) : 'N/A' },
    stellar_logg: { label: 'Surface Gravity', unit: 'log(g)', group: 'star', format: (v) => v ? v.toFixed(2) : 'N/A' },
    stellar_magnitude: { label: 'Stellar Magnitude', unit: 'mag', group: 'star', format: (v) => v ? v.toFixed(2) : 'N/A' },
    ra: { label: 'Right Ascension (RA)', group: 'star', format: (v) => v ? v.toFixed(4) : 'N/A' },
    dec: { label: 'Declination (Dec)', group: 'star', format: (v) => v ? v.toFixed(4) : 'N/A' },
};

// Helper function to dynamically generate HTML for info panels
export function createInfoRows(dataObject, group) {
    let html = '';
    // Display source separately, not in the loop
    const source = dataObject.source_telescope;
    if (source) {
        const sourceColor = source === 'TESS' ? 'text-cyan-400' : 'text-amber-400';
        html += `<p class="flex justify-between"><strong>Source:</strong> <span class="font-bold ${sourceColor}">${source}</span></p>`;
    }

    for (const key in parameterMap) {
        if (parameterMap[key].group === group && dataObject[key] !== null && dataObject[key] !== undefined) {
            const param = parameterMap[key];
            const rawValue = dataObject[key];
            const displayValue = param.format ? param.format(rawValue) : rawValue;
            const unit = param.unit ? ` ${param.unit}` : '';
            html += `<p class="flex justify-between"><strong>${param.label}:</strong> <span>${displayValue}${unit}</span></p>`;
        }
    }
    return html;
}
