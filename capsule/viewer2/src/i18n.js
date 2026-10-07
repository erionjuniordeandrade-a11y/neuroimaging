// Every interface string lives in this one pt-BR table. Static nodes use data-t="key".
const STRINGS={
  anonymized:'Anonimizado',pseudonymized:'Pseudonimizado',pseudonymizedTitle:'Identificadores removidos; o rosto continua visível na reconstrução 3D',offline:'Offline',intendedUse:'Apoio à discussão — não usar para navegação',loading:'Carregando cápsula',loadingVolume:'Preparando volume',loadingTract:'Carregando trato',
  empty:'Esta cápsula não contém volumes.',bad:'Não foi possível abrir a cápsula.',gpu:'WebGL2 indisponível neste navegador.',
  timeout:'A abertura está demorando mais que o esperado. Aguarde ou reabra o arquivo.',
  resampledReadout:'reamostrado de {source} mm para {grid} mm',
  noTour:'O cirurgião ainda não preparou a explicação deste caso.',step:'Passo',
  layout2x2:'2×2',layout3d:'3D',axial:'Axial',coronal:'Coronal',sagittal:'Sagital',
  patient:'Modo paciente',surgeon:'Modo cirurgião',save:'Salvar cápsula',saved:'Cápsula salva',
  base:'Base',overlay:'Sobreposição',none:'Nenhuma',colormap:'Cores',cmHot:'Quente',cmWarm:'Morno',cmCool:'Frio',opacity:'Opacidade',
  window:'Janela',level:'Nível',width:'Largura',layerBase:'Camada base',layerOverlay:'Camada sobreposta',manual:'Manual',auto:'Auto',
  render3d:'Renderização 3D',light:'Luz',clip:'Corte no cursor',off:'Desligado',resetView:'Redefinir vista',
  rotateHint:'Arraste para girar, role para ampliar.',viewHint:'Clique para mover o cursor · role para trocar de corte · arraste no 3D para girar',
  preset3dBone:'TC Osso',preset3dSkin:'TC Pele + osso',preset3dBrain:'RM Cérebro',preset3dSurfaces:'Superfícies (recorte)',preset3dSkinMR:'RM Pele',preset3dVessels:'RM Cérebro + vasos',preset3dCTA:'TC Angio (vasos + osso)',window3dMin:'Limiar 3D',window3dMax:'Teto 3D',notReviewed:'não revisado',preparing3d:'Preparando a visualização 3D',
  reviewer:'Revisor',reviewerPlaceholder:'Nome do revisor',reviewerRequired:'Informe o revisor antes de assinar a revisão.',reviewedUnsigned:'revisado (sem assinatura)',
  tracts:'Tratos',noTracts:'Nenhum trato nesta cápsula.',tractImportNote:'Tratos importados (.tck): calculados fora da cápsula',streamlines:'fibras',filteredShare:'filtradas',showOutliers:'Mostrar fibras filtradas',
  tractTrustRejected:'reprovado no QC de confiabilidade',tractTrustRatio:'tortuosidade {r}× a do lado oposto',tractTrustTortuous:'trajeto excessivamente tortuoso',tractTrustCap:'fibras no comprimento máximo',tractTrustRim:'passa junto à lesão',tractTrustYield:'poucas fibras: {r}× menos que o lado oposto (não prova ausência do trato)',tractTrustChip:'Feixe reprovado no QC',brainMaskQcFail:'Máscara cerebral reprovada no QC — render do cérebro indisponível',
  densityLabel:'Densidade:',densityProportional:'proporcional',densityEqualized:'igualada',densityWarning:'Densidade igualada — não compare lados pela aparência',
  asymmetryWarning:'Assimetria de contagem E/D ({counts}). Menos fibras não significa trato ausente: edema, ROI deslocada e limiar de FOD também reduzem a contagem.',
  tractProvenance:'Parâmetros de tractografia',tractProvenanceMissing:'Parâmetros de tractografia não registrados',
  slab:'Espessura no 2D (apenas visual)',tractXray:'Tratos através do cérebro',capsuleVersion:'versão',capsuleVersionTitle:'Versão desta cápsula (aumenta a cada salvamento)',
  masks:'Máscaras',noMasks:'Nenhuma máscara.',newMask:'Nova máscara',newMaskLabel:'Nova máscara',edit:'Editar',editing:'Editando',
  anatomy:'Anatomia',anatomyAuto:'Segmentação automática',opacity3d:'Opacidade no 3D',
  parcQcFail:'Parcelamento cortical falhou no controle de qualidade ({n} de 68 regiões discordantes) — não exibido',parcQcRejected:'Parcelamento reprovado no QC',qcUnavailable:'QC indisponível',qcMissing:'sem QC',qcPassFailed:'falhou QC',
  reviewed:'Revisado pelo cirurgião',show:'Mostrar',hide:'Ocultar',
  tools:'Ferramentas',toolCursor:'Cursor',toolDistance:'Distância',toolAngle:'Ângulo',toolPen:'Pincel +',toolErase:'Borracha',toolSegment:'Crescer região',toolTrajectory:'Trajetória',toolSegPrompt:'Pontos p/ IA',
  penSize:'Pincel',undo:'Desfazer',
  hintCursor:'Clique em qualquer corte para posicionar o cursor.',hintDistance:'Arraste num corte para medir a distância.',hintAngle:'Arraste a primeira linha e clique no terceiro ponto.',
  hintPen:'Pinte num corte para adicionar à máscara em edição.',hintErase:'Pinte num corte para apagar da máscara em edição.',hintSegment:'Clique numa estrutura para crescer a região (NiiVue).',
  hintSegPrompt:'Adicione uma estrutura pelo nome; clique no corte para ponto positivo, Shift+clique para negativo.',
  segPromptName:'Nome da estrutura',segPromptPlaceholder:'Ex.: Nervo facial',addSegPrompt:'Adicionar estrutura',
  segPromptPending:'Pendente',segPromptDone:'Concluída',segPromptEmpty:'Sem resultado',
  noSegPrompts:'Nenhuma estrutura adicionada.',segPromptNameRequired:'Digite o nome da estrutura.',
  segPromptNeedActive:'Adicione ou selecione uma estrutura pendente antes de marcar pontos.',
  hintTrajectory:'Clique no ponto de entrada e depois no alvo.',hintTrajectoryTarget:'Agora clique no alvo.',
  trajectory:'Trajetória',trajectoryLength:'Comprimento entrada → alvo',surgeonView:'Visão do cirurgião',trajClip:'Corte perpendicular à trajetória',trajDepth:'Profundidade',
  corridorRadius:'Raio do corredor',corridorHeader:'Estruturas a até {r} mm da trajetória, em ordem de profundidade a partir da entrada.',corridorEmpty:'Nenhuma estrutura segmentada no corredor.',
  corridorContact:'contato a',corridorCrosses:'atravessa a',corridorMin:'distância mínima',corridorGroupVessels:'Vasos',corridorGroupTracts:'Tratos',
  noAnnotations:'Nenhuma medida.',distance:'Distância',angle:'Ângulo',delete:'Excluir',
  tour:'Roteiro',noSteps:'Nenhum passo ainda.',addStep:'Adicionar passo',stepTitle:'Título do passo',stepText:'Explicação em português para o paciente',
  previous:'Anterior',next:'Próximo',patientSaveBlocked:'Salvamento indisponível no modo paciente',editBlocked:'Edição indisponível no modo paciente',
  readoutCursor:'Cursor'
};
function tr(key){return STRINGS[key]??key}
function formatVolumeMl(value){
  if(value===null||value===undefined||value==='')return '—';
  const number=Number(value);if(!Number.isFinite(number))return '—';
  return number.toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1})+' mL';
}
function localizeStatic(root=document){
  for(const el of root.querySelectorAll('[data-t]'))el.textContent=tr(el.dataset.t);
  $('new-step-title').placeholder=tr('stepTitle');$('new-step-text').placeholder=tr('stepText');$('seg-prompt-name').placeholder=tr('segPromptPlaceholder');
  $('reviewer-name').placeholder=tr('reviewerPlaceholder');
}
