import React from 'react';
import './App.css';
import { GraficoFocosPorAno } from './components/GraficoFocosPorAno';

function App() {
  return (
    <div className="App">
      <h1>Dashboard de Queimadas</h1>
      <GraficoFocosPorAno />
    </div>
  );
}

export default App;
