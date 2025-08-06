// src/components/GraficoFocosPorAno.tsx
import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { api } from '../services/api';

export function GraficoFocosPorAno() {
  const [dados, setDados] = useState([]);

  useEffect(() => {
    api.get('/queimadas/ano/2020')
      .then(response => setDados(response.data))
      .catch(error => console.error(error));
  }, []);

  return (
    <LineChart width={800} height={400} data={dados}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="data" />
      <YAxis />
      <Tooltip />
      <Line type="monotone" dataKey="qtd_focos" stroke="#ff7300" />
    </LineChart>
  );
}
