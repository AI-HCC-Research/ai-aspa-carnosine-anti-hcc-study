# -*- coding: utf-8 -*-
"""自绘 NAA / 4MXU过渡态类似物 / L-肌肽 三分子结构对比图 (RDKit, 无版权问题)"""
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

smiles = {
    'NAA\n(N-acetyl-L-aspartate)\nASPA sole natural substrate':
        'CC(=O)N[C@@H](CC(=O)O)C(=O)O',
    'N-phosphonomethyl-L-aspartate\nligand in PDB 4MXU\n(transition-state analog of NAA)':
        'O=P(O)(O)CN[C@@H](CC(=O)O)C(=O)O',
    'L-carnosine\n(beta-alanyl-L-histidine)\ncandidate ligand of this study':
        'O=C(O)[C@@H](Cc1c[nH]cn1)NC(=O)CCN',
}

mols = []
for label, smi in smiles.items():
    m = Chem.MolFromSmiles(smi)
    AllChem.Compute2DCoords(m)
    m.SetProp('_Name', label)
    mols.append(m)

img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(460, 380),
                           legends=list(smiles.keys()))
out = r'd:\ASPA-肌肽\文献调研_肌肽ASPA\figures\A0_三分子结构对比.png'
img.save(out, dpi=(200, 200))
print('saved', out)
