async function atualizarPainel() {
  const res = await fetch("/api/estatisticas");
  const dados = await res.json();

  document.getElementById("wins").innerText = dados.wins;
  document.getElementById("losses").innerText = dados.losses;
  document.getElementById("lucro_total").innerText = dados.lucro_total.toFixed(2);
  document.getElementById("status_bot").innerText = dados.ativo ? "Ativo" : "Desativado";

  const botao = document.getElementById("botao_toggle");
  botao.innerText = dados.ativo ? "Parar Bot" : "Iniciar Bot";

  const corpo = document.getElementById("trade_body");
  corpo.innerHTML = "";
  dados.ultimos_trades.forEach(t => {
    const linha = document.createElement("tr");
    linha.innerHTML = `
      <td>${t.timestamp}</td>
      <td>${t.direcao}</td>
      <td>$${t.stake}</td>
      <td>${t.resultado}</td>
      <td>$${t.lucro.toFixed(2)}</td>
    `;
    corpo.appendChild(linha);
  });
}

function salvarParametros() {
  const dados = {
    stake_inicial: parseFloat(document.getElementById("stake_inicial").value),
    stake_maxima: parseFloat(document.getElementById("stake_maxima").value),
    multiplicador: parseFloat(document.getElementById("multiplicador").value),
    limiar_ia: parseFloat(document.getElementById("limiar_ia").value),
  };

  fetch("/api/atualizar_parametros", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(dados)
  });
}

function resetarStats() {
  fetch("/api/reset", { method: "POST" }).then(() => atualizarPainel());
}

function toggleBot() {
  fetch("/api/estatisticas")
    .then(res => res.json())
    .then(data => {
      const ativo = data.ativo;
      const rota = ativo ? "/api/stop" : "/api/start";
      fetch(rota, { method: "POST" }).then(() => atualizarPainel());
    });
}

setInterval(atualizarPainel, 3000);
window.onload = atualizarPainel;
