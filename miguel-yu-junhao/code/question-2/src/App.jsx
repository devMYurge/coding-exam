import Medusae from './Medusae.jsx';

/**
 * The actual particle engine lives in ./Medusae.jsx — open it to read the
 * GLSL vertex / fragment shaders, the per-frame physics, the cursor logic.
 * Tunable defaults live in ./defaults.js; the `config` prop here overrides
 * them per-property (only `background` is overridden below; everything else
 * inherits the defaults, which already match the Antigravity reference).
 */
function App() {
  return (
    <Medusae
      config={{
        background: { color: '#ffffff' },
      }}
    />
  );
}

export default App;
