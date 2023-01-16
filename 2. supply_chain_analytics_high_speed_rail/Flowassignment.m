function [FlowStationTrainMatrix,FlowWarehouseStationMatrix,...
    FlowWarehouseWarehouseMatrix, FlowPlantWarehouseMatrix]...
    =Flowassignment(Route,FlowRoute,DecayPlaWarMatrix,...
    DecayWarWarMatrix,DecayWarStaMatrix,DecayStaTraMatrix)

[numberRoute,~]=size(Route);
[numberPlant,numberWarehouse]=size(DecayPlaWarMatrix);
[numberTrain,numberStation]=size(DecayStaTraMatrix);

FlowStationTrainMatrix=zeros(numberTrain,numberStation);
FlowWarehouseStationMatrix=zeros(numberWarehouse,numberStation);
FlowWarehouseWarehouseMatrix=zeros(numberWarehouse,numberWarehouse);
FlowPlantWarehouseMatrix=zeros(numberPlant,numberWarehouse);

% flow assignment considering flow decay on the links f_
for r=1:numberRoute
    FlowPlantWarehouseMatrix(Route(r,4),Route(r,3))=...
        FlowPlantWarehouseMatrix(Route(r,4),Route(r,3))+FlowRoute(r);
    Decay=DecayPlaWarMatrix(Route(r,4),Route(r,3));
    FlowWarehouseWarehouseMatrix(Route(r,3),Route(r,3))=...
        FlowWarehouseWarehouseMatrix(Route(r,3),Route(r,3))+FlowRoute(r)*Decay;
    Decay=Decay*DecayWarWarMatrix(Route(r,3),Route(r,3));
    FlowWarehouseStationMatrix(Route(r,3),Route(r,2))=...
        FlowWarehouseStationMatrix(Route(r,3),Route(r,2))+FlowRoute(r)*Decay;
    Decay=Decay*DecayWarStaMatrix(Route(r,3),Route(r,2));
    FlowStationTrainMatrix(Route(r,1),Route(r,2))=...
        FlowStationTrainMatrix(Route(r,1),Route(r,2))+FlowRoute(r)*Decay;
end

end