/* Buildless, dependency-free dashboard configuration. */
window.Monsoon = {
  config: {
    api: /^\/dashboard(?:\/|$)/.test(location.pathname) && /^https?:$/.test(location.protocol)
      ? location.origin : 'http://localhost:8000',
    height: 32, width: 36,
    channels: [
      {id: 't2m', name: 'Temperature', unit: 'K', colors: ['#243e69', '#478da8', '#99d4bc', '#f1cd7b', '#d96b50']},
      {id: 'tp', name: 'Precipitation', unit: 'mm/day', colors: ['#162b3b', '#225b78', '#3a9eaf', '#8dd5a8', '#f1d788']},
      {id: 'u10', name: 'Zonal wind', unit: 'm/s', colors: ['#356f9c', '#8bbfc8', '#d1d9d1', '#d8ad76', '#b66c55']},
      {id: 'v10', name: 'Meridional wind', unit: 'm/s', colors: ['#356f9c', '#8bbfc8', '#d1d9d1', '#d8ad76', '#b66c55']},
      {id: 'msl', name: 'Sea-level pressure', unit: 'Pa', colors: ['#344875', '#7179a6', '#afc2c6', '#e3c896', '#cc8e64']},
      {id: 'tcwv', name: 'Column water vapour', unit: 'kg/m²', colors: ['#443f64', '#486f99', '#4ca3af', '#95cebb', '#e0dc9d']}
    ],
    differenceColors: ['#5796db', '#a7c9e5', '#e1e4dc', '#dfa48c', '#e16d57'],
    presets: {
      bay: {channel: 0, magnitude: 2, lat: 16, lon: 88, description: 't2m +2 K · raised-cosine bump'},
      arabian: {channel: 2, magnitude: -3, lat: 16, lon: 70, description: 'u10 −3 m/s · raised-cosine bump'},
      stalled: {channel: 2, magnitude: -3, secondary: {channel: 5, ratio: 2}, lat: 20, lon: 75, description: 'u10 −3 m/s + tcwv −6 kg/m²'},
      heat: {channel: 0, magnitude: 4, lat: 26, lon: 78, description: 't2m +4 K · raised-cosine bump'}
    }
  }
};
