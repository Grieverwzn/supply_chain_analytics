function [IterOperationCost,IterDecayCost,DecayRouteMatrix,DecayRoute]=...
    OperationDiscardCostDev(ColumnPool,Capacity,...
    FixCostPlaWarMatrix,VarCostPlaWarMatrix,...
    DecayPlaWarMatrix,FlowPlantWarehouseMatrix,...
    FixCostWarWarMatrix,VarCostWarWarMatrix,DecayWarWarMatrix,...
    FlowWarehouseWarehouseMatrix,FixCostWarStaMatrix,...
    VarCostWarStaMatrix,DecayWarStaMatrix,FlowWarehouseStationMatrix,...
    FixCostStaTraMatrix,VarCostStaTraMatrix,DecayStaTraMatrix,...
    FlowStationTrainMatrix,PurchasePrice,WastePrice)


[numberRoute,~]=size(ColumnPool);
IterOperationCost=0*ones(1,numberRoute);
IterDecayCost=0*ones(1,numberRoute);
DecayRoute=0*ones(1,numberRoute);
%DecayRouteMatrix
% the first row: train id 
% the second row: station id 
% the third row: warehouse id 
% the fourth row: supplier/plant id 
DecayRouteMatrix=0*ones(4,numberRoute); % we will calculate the decay route matrix 

% for each route calculate its gradients of operational cost and discard
% cost (The first term and second term in the variational inequality)
for r=1:numberRoute
    Decay1=1;
    OperationCost=0;
    DecayCost=0;
    %-----------------supplier to warehouse---------------
    % ---------part 1: operational cost (nonlinear)--------- 
    % \hat{c}_(a)(f_{a}) = FC_{a} *f_{a}/cap_{a} + v(f_{a})*f_{a}
    % \hat{c}_{a}(f_{a}) = c_{a}(f_{a})*f_{a} % nonlinear function
    % c_{a}(f_{a}) = FC_{a} /cap_{a} + v(f_{a})
    % after the derivation: 
    % [c_{a}(f_{a})+ (\partial{c_{a}(f_{a})}{f_{a}})*f(a)]*Decay(a,p)
    % we can divide the function into two parts: 
    % 1. OO : c_{a}(f_{a}) = FC_{a} /cap_{a} + v(f_{a})
    % 2. OODev: (\partial{c_{a}(f_{a})}{f_{a}})*f(a)
    % since c_{a}(f_{a}) has two parts:  FC_{a} /cap_{a} is a constant
    % v(f_{a}) = v_{a}*f_{a}, after we take the derivative of c_{a}(f_{a})
    % we got v_{a}. 
    % Thus, OODev (\partial{c_{a}(f_{a})}{f_{a}})*f(a) = v_{a}*f_{a}
    % 3. Decay1 = 1 here,
    Flow= FlowPlantWarehouseMatrix(ColumnPool(r,4),ColumnPool(r,3));% ColumnPool 4 and 3 is the id of plants and warehouse 
    Decay=DecayPlaWarMatrix(ColumnPool(r,4),ColumnPool(r,3));
    DecayRouteMatrix(4,r)=Decay1; % the first layer no food decay 
    OO=Flow*(VarCostPlaWarMatrix(ColumnPool(r,4),ColumnPool(r,3)))+FixCostPlaWarMatrix(ColumnPool(r,4),ColumnPool(r,3))/Capacity(4);
    OODev= VarCostPlaWarMatrix(ColumnPool(r,4),ColumnPool(r,3))*Flow;
    OperationCostPlaWar=(OO+OODev)*Decay1;
    %---------part 2: discarding cost (linear)---------
    % % \hat{z}_{a}(f_{a}) = z_{a}*f_{a} % nonlinear function
    % z_{a} = (PurchasePrice-WastePrice)*(1-Decay); 1-Decay means the
    % products decayed or lost during th elink 
    % \partial{\hat{z}_{a}(f_{a}) }{f_{a}} = z_{a} = (PurchasePrice-WastePrice)*(1-Decay)
    % this is the reason why here we only have one term 
    DD=(PurchasePrice-WastePrice)*(1-Decay);%*Flow+(PurchasePrice-WastePrice)*(1-Decay);
    DecayCostPlaWar=DD*Decay1;
    %------------
    Decay1=Decay;
    %-----------------²Warehouse--------------------------
    Flow= FlowWarehouseWarehouseMatrix(ColumnPool(r,3),ColumnPool(r,3));
    Decay=DecayWarWarMatrix(ColumnPool(r,3),ColumnPool(r,3));
    DecayRouteMatrix(3,r)=Decay1;
    OO=Flow*(VarCostWarWarMatrix(ColumnPool(r,3),ColumnPool(r,3)))+FixCostWarWarMatrix(ColumnPool(r,3),ColumnPool(r,3))/Capacity(3);
    OODev= VarCostWarWarMatrix(ColumnPool(r,3),ColumnPool(r,3))*Flow;
    OperationCostWarWar=(OO+OODev)*Decay1;
    %-------------------------------------
    DD=(PurchasePrice-WastePrice)*(1-Decay);%*(PurchasePrice-WastePrice)*(1-Decay);
    DecayCostWarWar=DD*Decay1;
    %--------------------------------------
    Decay1=Decay*Decay1; % decay rate accumulation
    %-----------------Warehouse to station--------------------------  
    Flow= FlowWarehouseStationMatrix(ColumnPool(r,3),ColumnPool(r,2));
    Decay=DecayWarStaMatrix(ColumnPool(r,3),ColumnPool(r,2));
    DecayRouteMatrix(2,r)=Decay1;
    OO=Flow*(VarCostWarStaMatrix(ColumnPool(r,3),ColumnPool(r,2)))+FixCostWarStaMatrix(ColumnPool(r,3),ColumnPool(r,2))/Capacity(2);
    OODev= VarCostWarStaMatrix(ColumnPool(r,3),ColumnPool(r,2))*Flow;
    OperationCostWarSta=(OO+OODev)*Decay1;
    %------------------------------------------------
    DD=(PurchasePrice-WastePrice)*(1-Decay);%*Flow+(PurchasePrice-WastePrice)*(1-Decay);
    DecayCostWarSta=DD*Decay1;
    %--------------------------------------
    Decay1=Decay*Decay1; % decay rate accumulation
    %-----------------Station to train---------------
    Flow=FlowStationTrainMatrix(ColumnPool(r,1),ColumnPool(r,2));
    Decay=DecayStaTraMatrix(ColumnPool(r,1),ColumnPool(r,2));
    DecayRouteMatrix(1,r)=Decay1;
    OO=Flow*(VarCostStaTraMatrix(ColumnPool(r,1),ColumnPool(r,2)))+FixCostStaTraMatrix(ColumnPool(r,1),ColumnPool(r,2))/Capacity(1);
    OODev= VarCostStaTraMatrix(ColumnPool(r,1),ColumnPool(r,2))*Flow;
    OperationCostStaTra=(OO+OODev)*Decay1;
    %-------------------------------------------------
    DD=(PurchasePrice-WastePrice)*(1-Decay);%*Flow+(PurchasePrice-WastePrice)*(1-Decay);
    DecayCostStaTra=DD*Decay1;
    %--------------------------------------
    Decay1=Decay*Decay1; % decay rate accumulation
    %-----------------------------------Summation--------------------------------------
    OperationCost=OperationCostPlaWar+OperationCostWarWar+OperationCostWarSta+OperationCostStaTra;
    DecayCost=DecayCostPlaWar+DecayCostWarWar+DecayCostWarSta+DecayCostStaTra;
    IterOperationCost(r)=OperationCost; 
    % output 1: statistics of operation cost of each route 
    IterDecayCost(r)=DecayCost; 
    % output 2: statistics of discard cost of each route  
    DecayRoute(r)=Decay1; 
    % output 3: statistics of final decay rate  
    %of the route, for example Decay1=0.9 means 90% of products 
    %will be decayed from the supplier to the train
end

end