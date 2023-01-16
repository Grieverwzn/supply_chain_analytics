function [Langrange,RouteTimePool,PP,judge,RouteTimePool1]=...
    Multiplier(RouteFlow,Langrange,LagSTEP,LSpan,DurationStaTraMatrix,...
    ColumnPool,FlowPlantWarehouseMatrix,TimePlaWarMatrix,...
    FlowWarehouseWarehouseMatrix,TimeWarWarMatrix,...
    FlowWarehouseStationMatrix,TimeWarStaMatrix,...
    FlowStationTrainMatrix,TimeStaTraMatrix,Alpha,Belta,CapacityFacility)

[numberRoute,~]=size(ColumnPool);
RouteTimePool=zeros(1,numberRoute);
for r=1:numberRoute
    routeTime=0;
    %================supplier to warehouse===============
   Flow= FlowPlantWarehouseMatrix(ColumnPool(r,4),ColumnPool(r,3));
   Time=TimePlaWarMatrix(ColumnPool(r,4),ColumnPool(r,3));
   [T,~]=BPRDev(Flow,Time,Alpha(4),Belta(4),CapacityFacility(4)); 
   % the first output of BPRDev is just the BPR function 
   routeTime=routeTime+T;
   %=============warehouse===========================
   Flow= FlowWarehouseWarehouseMatrix(ColumnPool(r,3),ColumnPool(r,3));
   Time=TimeWarWarMatrix(ColumnPool(r,3),ColumnPool(r,3));
   [T,~]=BPRDev(Flow,Time,Alpha(3),Belta(3),CapacityFacility(3));
   % the first output of BPRDev is just the BPR function 
   routeTime=routeTime+T;
   %=============warehouse to station===========================
   Flow= FlowWarehouseStationMatrix(ColumnPool(r,3),ColumnPool(r,2));
   Time=TimeWarStaMatrix(ColumnPool(r,3),ColumnPool(r,2));
   [T,~]=BPRDev(Flow,Time,Alpha(2),Belta(2),CapacityFacility(2));
   % the first output of BPRDev is just the BPR function 
   routeTime=routeTime+T;
   %=============station to train===========================
   Flow= FlowStationTrainMatrix(ColumnPool(r,1),ColumnPool(r,2));
   Time=TimeStaTraMatrix(ColumnPool(r,1),ColumnPool(r,2));
   [T,~]=BPRDev(Flow,Time,Alpha(1),Belta(1),CapacityFacility(1));
   % the first output of BPRDev is just the BPR function 
   routeTime=routeTime+T;
   RouteTimePool(r)=routeTime;
end
%=============Gradient=================

RouteTimePool1=zeros(1,numberRoute);
judge=zeros(1,numberRoute);
Penality=zeros(1,numberRoute);
PP=zeros(1,numberRoute);
for r=1:numberRoute
    MM=DurationStaTraMatrix(ColumnPool(r,1),ColumnPool(r,2));
    %Duration in trains 
    %Langrange(r)=max(0,Langrange(r)+LagSTEP(r)*(RouteFlow(r)*RouteTimePool(r)+RouteFlow(r)*MM-(RouteFlow(r)*LSpan)));
    %if RouteFlow(r)>0
    Langrange(r)=max(0,Langrange(r)+LagSTEP(r)*(RouteTimePool(r)+MM-(LSpan)-DiracFunction(RouteFlow(r))));
    %end 
    %Langrange(r)=max(0,Langrange(r)+LagSTEP(r)*(RouteTimePool(r)+MM-(LSpan)));
    RouteTimePool1(r)=RouteTimePool(r)+MM;
    if RouteTimePool1(r)>LSpan+1
        judge(r)=1; % if route time > Lspan
    end
    % penalty values 
    PP(r)=max(0,RouteTimePool1(r)-LSpan-DiracFunction(RouteFlow(r)));    
    %end
end



end