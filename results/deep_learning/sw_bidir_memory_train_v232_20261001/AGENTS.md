Only14304 new norm/BiGRU/project params train; all original104 frozen. GRU float32/trainingmode for cudnnbackward,originalblocks eval/noDropKey. Originalfold5/Q50/score/80-14/max2GPU,oneSWaddition/no stacking/otherfolds. MaintainPROJECTS/evidence.

232 fresh version: explicitly set only new recurrent module training flag after parent.eval();231 intended switch was absent,firstbackwardfailed/no epochs. Keep231source/config/error immutable,232 from104not231weights. Evaluation recurrentflagFalse; originalblocks alwayseval.
