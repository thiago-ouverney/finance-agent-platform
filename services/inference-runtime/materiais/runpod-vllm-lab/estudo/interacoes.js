'use strict';
const $ = id => document.getElementById(id);
const n = id => Number($(id).value);
const fmt = (value, digits = 1) => new Intl.NumberFormat('pt-BR', {maximumFractionDigits: digits, minimumFractionDigits: digits}).format(value);
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const colors = {green:'#176956', orange:'#b65327', blue:'#3b619c', violet:'#7151a0', ink:'#172e31'};
const storage = {
  get(key) {try {return localStorage.getItem('yan-estudo-v1-' + key);} catch {return null;}},
  set(key, value) {try {localStorage.setItem('yan-estudo-v1-' + key, value);return true;} catch {return false;}}
};
function bind(ids, callback) {ids.forEach(id => $(id).addEventListener('input', callback)); callback();}
function metric(value, label) {return `<div class="metric"><strong>${value}</strong>${label}</div>`;}
function out(id, text) {$(id + '-o').textContent = text;}

// Cada cálculo fica numa função pura para ser auditado/testado separadamente.
function timeline(queue, prefill, tpot, tokens) {
  const ttft = queue + prefill;
  const decode = (tokens - 1) * tpot;
  return {ttft, decode, total:ttft + decode, rate:1000 / tpot};
}
function roofline(compute, bandwidth, intensity) {
  return {bound:Math.min(compute, bandwidth * intensity / 1000), knee:compute * 1000 / bandwidth, memory:bandwidth * intensity / 1000 < compute};
}
function kvBudget(weights, overhead, context, sequences, util) {
  const kv = 2 * 48 * 8 * 128 * 2 * context * sequences / 2**30;
  const budget = 24 * util;
  const total = weights + overhead + kv;
  return {kv,budget,total,remaining:budget-total};
}
function schedule(chunk) {
  let remain=1024, elapsed=0;const rounds=[];
  while(remain>0) {const tokens=Math.min(remain,chunk);const time=10+tokens*.1;rounds.push({tokens,time,start:elapsed});elapsed+=time;remain-=tokens;}
  return {rounds,total:elapsed,maxGap:Math.max(...rounds.map(x=>x.time))};
}
// Exposição apenas para testes locais dos modelos didáticos.
window.studyModels = {timeline,roofline,kvBudget,schedule};

const stages = [
  '<strong>Navegador → servidor.</strong> O texto e o histórico atravessam a rede e o proxy do RunPod. Demora percebida aqui pode existir mesmo se a GPU estiver rápida. Para isolar execução, meça também com o cliente dentro do Pod.',
  '<strong>CPU + RAM.</strong> A API recebe a requisição, aplica o template e tokeniza a conversa. Texto vira IDs; não são os 14 bilhões de pesos atravessando a rede a cada pergunta. A CPU também coordena o restante do trabalho.',
  '<strong>Agendador.</strong> Uma requisição pode esperar enquanto outras ocupam recursos. O runtime decide quais sequências entram na próxima iteração e quais blocos de cache precisam estar disponíveis. Fila e execução são tempos diferentes.',
  '<strong>GPU + VRAM.</strong> Pesos, ativações e KV cache alimentam os kernels. O cálculo acontece na GPU; ler VRAM também custa tempo. Com pesos residentes, o caminho de leitura principal não é SSD → PCIe a cada token.',
  '<strong>Streaming.</strong> Tokens viram texto e chegam em eventos ao cliente. Um evento pode conter vários tokens ou apenas metadados. O primeiro evento HTTP não necessariamente marca o primeiro token de conteúdo. Renderização também afeta a sensação de fluidez.'
];
function showStage(index) {$('stage-text').innerHTML=stages[index];document.querySelectorAll('[data-stage]').forEach(b=>b.setAttribute('aria-pressed', String(Number(b.dataset.stage)===index)));}
document.querySelectorAll('[data-stage]').forEach(button=>button.addEventListener('click',()=>showStage(Number(button.dataset.stage))));showStage(0);

bind(['queue','prefill','tpot','outtokens'],()=>{
  const q=n('queue'),p=n('prefill'),d=n('tpot'),count=n('outtokens');const t=timeline(q,p,d,count);
  out('queue',fmt(q,0)+' ms');out('prefill',fmt(p,0)+' ms');out('tpot',fmt(d,0)+' ms/token');out('outtokens',count+' tokens');
  $('time-bar').innerHTML=[[q,colors.orange],[p,colors.blue],[t.decode,colors.green]].map(([duration,color])=>`<span style="width:${100*duration/t.total}%;background:${color}"></span>`).join('');
  $('time-metrics').innerHTML=metric(fmt(t.ttft/1000,2)+' s','TTFT: espera inicial')+metric(fmt(t.rate,1)+' tok/s','decode após o primeiro token')+metric(fmt(t.total/1000,2)+' s','tempo total da resposta');
  $('time-insight').textContent=`A fila representa ${fmt(q/t.total*100,1)}% do total. A taxa que inclui toda a espera seria ${fmt(count/(t.total/1000),1)} tok/s, diferente dos ${fmt(t.rate,1)} tok/s de decode. Trocar somente a fila não muda o ritmo de geração neste modelo.`;
});

function roofPlot(c,b,i){
  const x=v=>65+Math.log10(v)/3*690;
  const y=v=>300-(Math.log10(v)+1)/(Math.log10(250)+1)*255;
  let svg='<title>Limite Roofline num hardware hipotético</title><desc>Intensidade no eixo horizontal, taxa de cálculo no vertical. Ambos em escala logarítmica.</desc>';
  for(const v of [1,10,100,1000])svg+=`<line class="grid" x1="${x(v)}" y1="35" x2="${x(v)}" y2="300"/><text x="${x(v)}" y="322" text-anchor="middle">${v}</text>`;
  for(const v of [.1,1,10,100])svg+=`<line class="grid" x1="65" y1="${y(v)}" x2="755" y2="${y(v)}"/><text x="55" y="${y(v)+4}" text-anchor="end">${fmt(v,v<1?1:0)}</text>`;
  svg+='<text x="65" y="20">TFLOP/s · log</text><text x="410" y="350" text-anchor="middle">Intensidade aritmética (operações/byte) · log</text>';
  const points=Array.from({length:181},(_,j)=>{const intensity=10**(j/60);return `${x(intensity)},${y(roofline(c,b,intensity).bound)}`}).join(' ');
  const r=roofline(c,b,i);
  svg+=`<polyline points="${points}" fill="none" stroke="${colors.green}" stroke-width="4"/><line x1="${x(i)}" y1="300" x2="${x(i)}" y2="${y(r.bound)}" stroke="${colors.orange}" stroke-dasharray="5 4"/><circle cx="${x(i)}" cy="${y(r.bound)}" r="7" fill="${colors.orange}" stroke="white" stroke-width="2"/>`;
  if(r.knee>=1&&r.knee<=1000)svg+=`<line x1="${x(r.knee)}" y1="35" x2="${x(r.knee)}" y2="300" stroke="#819b8a" stroke-dasharray="3 6"/><text x="${Math.min(x(r.knee)+8,665)}" y="50">joelho: ${fmt(r.knee,1)}</text>`;
  return svg;
}
function updateRoof(){const c=n('compute'),b=n('bandwidth'),i=10**n('intensity'),r=roofline(c,b,i);out('compute',fmt(c,0)+' TFLOP/s');out('bandwidth',fmt(b,0)+' GB/s');out('intensity',fmt(i,1)+' op/byte');$('roof-plot').innerHTML=roofPlot(c,b,i);$('roof-metrics').innerHTML=metric(fmt(r.bound,2)+' TFLOP/s','limite do modelo')+metric(fmt(r.knee,1)+' op/byte','intensidade no joelho')+metric(r.memory?'Banda':'Cálculo','recurso limitante no modelo');$('roof-insight').textContent=r.memory?'Abaixo do joelho: o teto de banda está abaixo do teto de cálculo. Aumentar só C não melhora esse limite; aumentar reuso ou banda pode melhorá-lo.':'No joelho ou à direita: a memória já permite chegar ao teto de cálculo deste modelo. Mais banda, sozinha, não eleva o limite atual.';}
bind(['compute','bandwidth','intensity'],updateRoof);
$('roof-decode').addEventListener('click',()=>{$('intensity').value=Math.log10(2);updateRoof();});
$('roof-reuse').addEventListener('click',()=>{$('intensity').value=Math.log10(256);updateRoof();});

bind(['transfer-size','pcie-band','vram-band'],()=>{
  const size=n('transfer-size'),pcie=n('pcie-band'),vram=n('vram-band');out('transfer-size',fmt(size,2)+' GB');out('pcie-band',fmt(pcie,0)+' GB/s');out('vram-band',fmt(vram,0)+' GB/s');
  $('transfer-metrics').innerHTML=metric(fmt(1000*size/pcie,1)+' ms','tempo puro na conexão CPU–GPU')+metric(fmt(1000*size/vram,2)+' ms','mesmo volume pela interface VRAM')+metric(size?fmt(pcie/size,1)+' passos/s':'—','inverso do tempo de transferência');
  $('transfer-insight').textContent=size?`Para esses valores, a transferência CPU–GPU leva ${fmt(vram/pcie,1)} vezes o tempo da interface local. Se cada passo precisa desses bytes, escondê-los completamente exige trabalho independente suficiente para sobrepor.`:'Com zero bytes nessa conexão, o modelo não impõe limite por esse tráfego. Isso não significa inferência instantânea: cálculo e acessos à VRAM continuam existindo.';
});

bind(['weights','overhead','context','sequences','util'],()=>{
  const w=n('weights'),o=n('overhead'),t=n('context'),s=n('sequences'),u=n('util');const r=kvBudget(w,o,t,s,u);
  out('weights',fmt(w,2)+' GiB');out('overhead',fmt(o,2)+' GiB');out('context',fmt(t,0)+' tokens');out('sequences',String(s));out('util',fmt(u*100,0)+'%');
  let remaining=24;const segments=[[w,colors.green],[o,colors.violet],[r.kv,colors.blue],[24-r.budget,colors.orange]];
  $('kv-bar').innerHTML=segments.map(([value,color])=>{const shown=Math.min(value,Math.max(remaining,0));remaining-=shown;return `<span style="width:${shown/24*100}%;background:${color}"></span>`;}).join('');
  $('kv-metrics').innerHTML=metric(fmt(r.kv,2)+' GiB','KV lógico necessário')+metric(fmt(r.budget,2)+' GiB','orçamento do motor')+metric(fmt(r.remaining,2)+' GiB','folga no orçamento');
  $('kv-insight').textContent=r.remaining>=0?`A demanda estimada é ${fmt(r.total,2)} GiB e cabe neste orçamento simplificado. Isso é condição de capacidade, não garantia de velocidade nem de startup. A pré-alocação do pool pode deixar a placa aparentemente cheia.`:`A demanda estimada é ${fmt(r.total,2)} GiB, acima do orçamento em ${fmt(-r.remaining,2)} GiB. O gráfico é cortado no total físico de 24 GiB. Reduza tokens/concorrência ou reveja o orçamento; elevar a fração não cria memória física.`;
});

function pages(demand){
  const blocks=[2,3,1], letters=['A','B','C'];let cells=[];
  blocks.forEach((used,i)=>{for(let j=0;j<(demand?used:4);j++)cells.push(`<span class="memory-cell ${j<used?letters[i].toLowerCase():'waste'}" title="${j<used?'Bloco da conversa '+letters[i]:'Reserva sem dados'}">${j<used?letters[i]:''}</span>`);});
  while(cells.length<12)cells.push('<span class="memory-cell" title="Bloco livre"></span>');
  $('pages-grid').innerHTML=cells.join('');$('pages-fixed').setAttribute('aria-pressed',String(!demand));$('pages-demand').setAttribute('aria-pressed',String(demand));
  $('pages-insight').textContent=demand?'6 blocos alocados = 24 posições para 17 tokens. Restam 7 posições internas sem uso e 6 blocos livres. Na implementação real, blocos de uma sequência podem estar espalhados; a maquete os agrupa só para facilitar a leitura.':'12 blocos reservados = 48 posições para 17 tokens. Há 31 posições ainda sem dados, incluindo 6 blocos inteiros hachurados. Reservar pelo máximo ocupa espaço antes de precisar dele.';
}
$('pages-fixed').addEventListener('click',()=>pages(false));$('pages-demand').addEventListener('click',()=>pages(true));pages(false);

bind(['chunk'],()=>{
  const c=n('chunk'),r=schedule(c);out('chunk',c+' tokens');let svg='<title>Agendamento simplificado de duas conversas</title><text x="20" y="25">A · tokens de decode ao fim de cada rodada</text>';
  const x=t=>30+t/r.total*730;
  r.rounds.forEach((round,index)=>{const x0=x(round.start),x1=x(round.start+10),x2=x(round.start+round.time);svg+=`<rect x="${x0}" y="95" width="${x1-x0}" height="32" fill="${colors.green}"/><rect x="${x1}" y="95" width="${x2-x1}" height="32" fill="${colors.blue}"/><circle cx="${x2}" cy="53" r="5" fill="${colors.green}"/><line x1="${x2}" y1="62" x2="${x2}" y2="132" stroke="#b7cabc" stroke-dasharray="3 3"/>`;if(r.rounds.length<=8)svg+=`<text x="${(x0+x2)/2}" y="150" text-anchor="middle">${index+1}</text>`;});
  svg+='<text x="20" y="180">Verde: custo fixo com decode · azul: prefill · eixo normalizado ao tempo total da simulação</text>';
  $('schedule-plot').innerHTML=svg;$('schedule-metrics').innerHTML=metric(r.rounds.length,'rodadas para ler B')+metric(fmt(r.maxGap,1)+' ms','maior intervalo entre entregas de A')+metric(fmt(r.total,1)+' ms','até terminar o prefill de B');
  $('schedule-insight').textContent=`Blocos de ${c} tokens produzem ${r.rounds.length} rodadas. Blocos menores reduzem a maior pausa de A, mas repetem o custo fixo mais vezes antes de B terminar o prefill. Isso demonstra um trade-off; estes milissegundos não são um benchmark.`;
});

const params={
  context:{name:'--max-model-len',meaning:'Limite aceito de entrada + saída por sequência.',why:'Permitir o contexto necessário ao chatbot e tornar a demanda previsível.',test:'Mantenha o prompt real igual e compare limites 2048 e 4096, se ambos inicializarem. Separadamente, aumente o prompt real: são intervenções diferentes.',measure:'Aceitação de entradas, memória na inicialização e latência com comprimentos reais registrados.',trap:'Aumentar o teto não torna a pergunta atual mais longa. Diminuí-lo não significa que o pool KV encolherá proporcionalmente.',source:'https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/model.py'},
  memory:{name:'--gpu-memory-utilization',meaning:'Fração de memória da GPU usada no orçamento do executor, não fração de poder computacional.',why:'Dar espaço ao cache quando ele limita execução, mantendo margem para o ambiente.',test:'Após uma referência estável, contraste 0.85 e 0.90. Se 0.85 nem carregar, registre falha; não invente latência para essa condição.',measure:'Capacidade de cache anunciada, preempções, OOM, fila e latência.',trap:'Não é um multiplicador de velocidade. 0.95 pode piorar a margem e 1.0 não é um alvo automático. A classe de cache v0.29.0 declara 0.92; confira o valor resolvido na sua instalação.',source:'https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/cache.py'},
  seqs:{name:'--max-num-seqs',meaning:'Limite de sequências processadas numa iteração do agendador.',why:'Estudar capacidade de agrupar trabalho de várias conversas.',test:'Contraste 1, 2 e 4 com um cliente que realmente envie carga concorrente. Registre também a taxa de chegada.',measure:'TTFT/TPOT por usuário, vazão agregada, espera e demanda KV.',trap:'Limite 4 não significa batch sempre igual a 4. Não significa também quatro cópias independentes dos pesos na GPU.',source:'https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/scheduler.py'},
  batch:{name:'--max-num-batched-tokens',meaning:'Orçamento de tokens processados por iteração, incluindo trabalho de várias sequências.',why:'Estudar como prefills e decodes disputam rodadas de execução.',test:'Com chunked prefill habilitado e carga mista, contraste 512, 1024 e 2048. Valide os argumentos na versão instalada.',measure:'Pausas durante o decode de uma conversa e TTFT da pergunta longa que chega.',trap:'Não é max_tokens da resposta. Sem chunking, o orçamento precisa comportar o contexto conforme as restrições da versão. Não o reduza cegamente.',source:'https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/scheduler.py'},
  prefix:{name:'--enable-prefix-caching / --no-enable-prefix-caching',meaning:'Permite ou desativa reuso de blocos KV de prefixos já processados.',why:'Medir a vantagem de contexto compartilhado ou histórico repetido.',test:'Compare o mesmo prefixo longo com perguntas finais novas. Separe primeira utilização de reuso.',measure:'TTFT e taxa de acerto de prefixo quando disponível.',trap:'Cache hit exige prefixos tokenizados compatíveis, não semelhança semântica. Não é um cache da resposta pronta.',source:'https://docs.vllm.ai/en/v0.29.0/features/automatic_prefix_caching/'},
  eager:{name:'--enforce-eager',meaning:'Força o caminho eager, desabilitando CUDA graphs para a execução do modelo.',why:'Investigar custos de submissão/execução e problemas de preparação dos graphs.',test:'Compare execução padrão com a mesma configuração acrescida dessa flag. Faça aquecimento em cada condição.',measure:'Startup separado de TTFT, TPOT e memória aquecidos.',trap:'Inicializar mais rápido não significa responder mais rápido. A flag é um contraste diagnóstico, não recomendação universal.',source:'https://docs.vllm.ai/en/v0.29.0/design/cuda_graphs/'},
  kv:{name:'--kv-cache-dtype',meaning:'Precisão dos elementos do KV cache; é distinta da quantização dos pesos.',why:'Só explorar depois, se o estado das conversas consumir a capacidade necessária.',test:'Mantenha auto na primeira investigação. Antes de trocar, confirme suporte de hardware/backend e a necessidade de escalas.',measure:'Capacidade, latência e qualidade/contexto das respostas.',trap:'Q8_0 nos pesos não implica KV em 8 bits. Mudar precisão muda a aproximação numérica e exige checagem de regressão.',source:'https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/cache.py'},
  offload:{name:'Offload de pesos para CPU',meaning:'Mantém parte de pesos fora da VRAM, com acesso pela conexão CPU–GPU.',why:'Viabilizar capacidade ou estudar deliberadamente o custo do caminho de memória.',test:'Deixe fora da trilha inicial se o checkpoint cabe. A interface de configuração é sensível à versão: consulte --help=all e a configuração de offload antes de experimentar.',measure:'RAM, tráfego/tempo de transferência, tempo de execução e contexto de cada teste.',trap:'RAM não se soma à VRAM com a mesma velocidade. O argumento de espaço não determina sozinho quantos bytes são transferidos em cada passo.',source:'https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/offload.py'}
};
function updateParam(){const p=params[$('param-choice').value];$('param-detail').innerHTML=`<h3><code>${p.name}</code></h3><p>${p.meaning}</p><div class="cols"><div><h4>Quando investigar</h4><p>${p.why}</p><h4>Um contraste possível</h4><p>${p.test}</p></div><div><h4>O que medir</h4><p>${p.measure}</p><h4>Armadilha</h4><p>${p.trap}</p></div></div><p class="source"><a href="${p.source}">Definição / mecanismo na fonte oficial ↗</a></p>`;}
$('param-choice').addEventListener('change',updateParam);updateParam();

const diagnoses={
  idle:'<strong>Hipótese inicial: reserva normal de memória.</strong> Veja se o vLLM pré-alocou o pool KV e se a GPU está ociosa por ausência de trabalho. Compare uso lógico de cache e contadores de requisições. Não conclua que há vazamento só porque os gigabytes não caem após uma resposta.',
  first:'<strong>Separe startup de requisição.</strong> Download, leitura de pesos, compilação e captura de graphs podem dominar o primeiro início. Faça uma medição aquecida; não use tempo de download como TTFT do serviço pronto.',
  prompt:'<strong>Hipótese: mais trabalho de prefill.</strong> Fixe saída e concorrência, desligue/controle reuso de prefixos e meça várias entradas. Confira fila e tokenização. O comportamento é compatível com prefill mais caro, mas não identifica sozinho qual recurso físico o limita.',
  decode:'<strong>Não conclua compute-bound.</strong> Consulte o Roofline para formular hipóteses de reuso. Verifique offload, histórico e kernels. Um profiler de memória/cálculo é mais conclusivo que GPU-Util; batch diferente também altera outras partes do sistema.',
  cpu:'<strong>Hipótese: alimentação/coordenação limitada por CPU.</strong> Observe threads, quota e throttling. Evite outros processos na medição e compare preparação com execução. GPU esperando também pode ser falta de requisições, bloqueio de I/O ou sincronização.',
  queue:'<strong>Separe fila de tempo de serviço.</strong> Compare requisições waiting/running, taxa real de chegada e preempções. Reduza carga mantendo os mesmos casos. Se a espera cair, há evidência de pressão de carga; identifique depois por que a capacidade de atendimento é limitada.',
  oom:'<strong>Qual memória acabou?</strong> CUDA OOM aponta para alocação na GPU; Killed sem traceback pode ser OOM de RAM/container. Leia o log e diferenças em memory.events. OOM na captura de graphs, no carregamento dos pesos e durante contexto longo são estágios diferentes.'
};
function diagnosis(){$('diagnosis').innerHTML=diagnoses[$('symptom').value];}$('symptom').addEventListener('change',diagnosis);diagnosis();

const questions=[
  {q:'1. O Pod tem 64 GB de RAM e 24 GB de VRAM. Tenho 88 GB com a velocidade da memória da GPU?',answers:['Sim, o vLLM soma automaticamente as duas.','Não; são espaços distintos, e acessos entre eles têm custo.'],correct:1,explanation:'Capacidades podem ajudar a viabilizar uma execução com offload, mas a conexão e a memória acessada mudam. O orçamento de VRAM continua sendo uma restrição própria.'},
  {q:'2. Quando um artigo diz “ler pesos da HBM”, como isso se aplica à 3090?',answers:['A 3090 também usa HBM, pois toda VRAM é HBM.','A ideia é leitura da memória local da GPU; a tecnologia da 3090 é GDDR6X.'],correct:1,explanation:'VRAM é o papel da memória local da GPU; HBM e GDDR6X são tecnologias diferentes. Não copie a banda de uma A100 para o modelo da 3090.'},
  {q:'3. GPU-Util em 99% comprova que o cálculo é o gargalo?',answers:['Não; atividade não mede diretamente saturação de FLOPS ou banda.','Sim, pois 99% significa 99% dos FLOPS de pico.'],correct:0,explanation:'Um kernel pode estar ativo e limitado por memória, dependências ou outros custos. Precisamos de evidência adicional do mecanismo.'},
  {q:'4. A mesma pergunta fica muito mais rápida na terceira tentativa. O runtime foi otimizado?',answers:['Pode haver cache ou aquecimento; isso precisa ser controlado.','Sim, três repetições bastam para demonstrar ganho.'],correct:0,explanation:'Sem controlar prefix cache e preparação inicial, as tentativas não representam a mesma condição. O reuso é útil no chatbot, mas deve ser declarado.'},
  {q:'5. PagedAttention significa que o KV cache vai para o SSD?',answers:['Sim; toda paginação é swap.','Não; blocos do KV podem continuar inteiramente na VRAM.'],correct:1,explanation:'A analogia é de organização e mapeamento de blocos, não uma obrigação de mover dados para disco. Swap do SO e preempção do runtime são mecanismos distintos.'},
  {q:'6. Dobrei throughput agregado, mas cada usuário recebe menos tokens/s. É possível?',answers:['Sim, atender mais sequências pode elevar a soma e reduzir a taxa individual.','Não; as duas métricas são equivalentes.'],correct:0,explanation:'Por isso a avaliação do chatbot inclui TTFT e decode por requisição, além da vazão agregada.'},
  {q:'7. Aumentei max-model-len de 2048 para 4096, mantendo a mesma pergunta curta. Dobrei o prefill?',answers:['Sim, o servidor sempre preenche todo o limite.','Não; o comprimento efetivo depende da entrada tokenizada.'],correct:1,explanation:'O limite aceito e o tamanho real são variáveis diferentes. O teto pode influenciar preparação/capacidade, mas não acrescenta palavras à sua pergunta.'}
];
$('quiz').innerHTML=questions.map((q,i)=>`<div class="card" style="margin:18px 0"><h3>${q.q}</h3>${q.answers.map((a,j)=>`<button class="quiz-choice" data-q="${i}" data-a="${j}">${escapeHtml(a)}</button>`).join('')}<div id="quiz-${i}" aria-live="polite"></div></div>`).join('');
document.querySelectorAll('[data-q]').forEach(button=>button.addEventListener('click',()=>{const q=questions[Number(button.dataset.q)],good=Number(button.dataset.a)===q.correct;const target=$('quiz-'+button.dataset.q);target.className='quiz-result '+(good?'good':'bad');target.textContent=(good?'Isso. ':'Reveja esta distinção: ')+q.explanation;}));

$('notes').value=storage.get('notes')||'';
$('save-notes').addEventListener('click',()=>{$('notes-status').textContent=storage.set('notes',$('notes').value)?'Nota salva neste navegador. Exporte uma cópia para manter seu registro.':'Este navegador não permitiu salvar localmente. Use Baixar nota .md.';});
$('download-notes').addEventListener('click',()=>{const text='# Caderno de hipótese — Yan\n\n'+new Date().toISOString()+'\n\n'+$('notes').value+'\n';const url=URL.createObjectURL(new Blob([text],{type:'text/markdown;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='hipotese-vllm.md';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);$('notes-status').textContent='Download solicitado. Confira a pasta de downloads.';});

document.querySelectorAll('pre').forEach(pre=>{const wrap=document.createElement('div');wrap.className='code-wrap';pre.before(wrap);wrap.append(pre);const button=document.createElement('button');button.textContent='Copiar';button.type='button';button.setAttribute('aria-label','Copiar bloco de comandos');wrap.append(button);button.addEventListener('click',async()=>{let ok=false;try{await navigator.clipboard.writeText(pre.textContent);ok=true;}catch{const input=document.createElement('textarea');input.value=pre.textContent;input.style.position='fixed';input.style.opacity='0';document.body.append(input);input.select();try{ok=document.execCommand('copy');}catch{}input.remove();}button.textContent=ok?'Copiado':'Selecione o texto';$('global-status').textContent=ok?'Comandos copiados.':'Selecione e copie o bloco manualmente.';setTimeout(()=>button.textContent='Copiar',1800);});});

const chapters=[...document.querySelectorAll('section.chapter:not(#fontes)')];
chapters.forEach(section=>{const label=document.createElement('label');label.className='chapter-done';const input=document.createElement('input');input.type='checkbox';input.checked=storage.get('done-'+section.id)==='yes';label.append(input,document.createTextNode('Li este capítulo e experimentei os controles.'));section.append(label);input.addEventListener('change',()=>{storage.set('done-'+section.id,input.checked?'yes':'no');progress();});});
function progress(){const done=document.querySelectorAll('.chapter-done input:checked').length;$('progress').style.width=100*done/chapters.length+'%';$('progress-note').textContent=`${done} de ${chapters.length} capítulos marcados`;}
progress();$('print-page').addEventListener('click',()=>window.print());
const nav=document.querySelector('aside nav');$('mobile-nav').innerHTML=nav.innerHTML;
if('IntersectionObserver' in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+entry.target.id));}}},{rootMargin:'-5% 0px -70% 0px',threshold:0});document.querySelectorAll('section.chapter').forEach(section=>observer.observe(section));}
