## demo.py

Matrices encode a biological knowledge base (genome-scale metabolic network), iML1515 [1]. We used this model in our paper [2] with [SWI-Prolog matrix encoding](https://github.com/lun-ai/BMLP). This demo shows the GPU version can replace the SWI-Prolog implementation.

Ran 2 tests on BMLP-IE with the GPU version given a conversion from SWI-Prolog input boolean matrices.

## large_net.py

It shows larger network examples rather than in notebook due to memory constraints.

## References

[1] Monk, J., C. Lloyd, and others. ‘iML1515, a Knowledgebase That Computes Escherichia Coli Traits’. Nature Biotechnology 35 (October 2017): 904–8. https://doi.org/10.1038/nbt.3956.

[2] Ai, Lun, and Stephen H. Muggleton. ‘Boolean Matrix Logic Programming’. arXiv, 19 August 2024. https://doi.org/10.48550/arXiv.2408.10369.