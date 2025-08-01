function [GeneralizedCost]=...
    GeneralizedDeadlineCost(Langrange,ColumnPool,DecayRouteMatrix,...
    FlowPlantWarehouseMatrix, TimePlaWarMatrix,...
    FlowWarehouseWarehouseMatrix, TimeWarWarMatrix,...
    FlowWarehouseStationMatrix, TimeWarStaMatrix,...
    FlowStationTrainMatrix, TimeStaTraMatrix,Alpha,Belta,CapacityFacility) 

[numberRoute,~]=size(ColumnPool);
GeneralizedCost=0*ones(1,numberRoute);
%parallel program
% PPP=zeros(1,numberRoute);
% for r=1:numberRoute
%     PPP(r)=RouteTimePool1(r)-LSpan;
% end

% time-deadline constraints: T_{q}(x) + duration in trains <=LSpan+dirac
% take derivative of the constraints 
% \partial{T_{q}(x)}{x_{p}}
% because: 
% 1. duration in trains is a constant 
% 2. Lspan is a constant 
% 3. we assume that dirac delta has derivative = 0 approximately
% 4. \partial{T_{q}(x)}{x_{p}} = sum_{a}{BPRdev(a)*incidence{a,q}*Decay(a,p)}

for r=1:numberRoute
    %---------for each path we should add derivatives of all others path's timedeadline constraints---------------
    AdditionCost=0;
    for k=1:numberRoute
        lang=Langrange(k);
        GC=0; % generalized cost 
        %================plants to warehouse===============
        if ColumnPool(r,4)==ColumnPool(k,4) && ColumnPool(r,3)==ColumnPool(k,3)
        Flow= FlowPlantWarehouseMatrix(ColumnPool(k,4),ColumnPool(k,3));
        Time=TimePlaWarMatrix(ColumnPool(k,4),ColumnPool(k,3));
        [~,TT]=BPRDev(Flow,Time,Alpha(4),Belta(4),CapacityFacility(4));
        GC=GC+TT*ceil(DecayRouteMatrix(4,k))*DecayRouteMatrix(4,r);
        % incidence(a,p) = ceil(DecayRouteMatrix(4,k)) (inputs are 0 or 1)
        end
        %=============warehouse===========================
        if ColumnPool(r,3)==ColumnPool(k,3) 
        Flow= FlowWarehouseWarehouseMatrix(ColumnPool(k,3),ColumnPool(k,3));
        Time=TimeWarWarMatrix(ColumnPool(k,3),ColumnPool(k,3));
        [~,TT]=BPRDev(Flow,Time,Alpha(3),Belta(3),CapacityFacility(3));
        GC=GC+TT*ceil(DecayRouteMatrix(3,k))*DecayRouteMatrix(3,r);
        % incidence(a,p) = ceil(DecayRouteMatrix(4,k)) (inputs are 0 or 1)
        end
        %=============warehouse to station===========================
        if  ColumnPool(r,3)==ColumnPool(k,3) && ColumnPool(r,2)==ColumnPool(k,2)
        Flow= FlowWarehouseStationMatrix(ColumnPool(k,3),ColumnPool(k,2));
        Time=TimeWarStaMatrix(ColumnPool(k,3),ColumnPool(k,2));
        [~,TT]=BPRDev(Flow,Time,Alpha(2),Belta(2),CapacityFacility(2));
        GC=GC+TT*ceil(DecayRouteMatrix(2,k))*DecayRouteMatrix(2,r);
        % incidence(a,p) = ceil(DecayRouteMatrix(4,k)) (inputs are 0 or 1)
        end
        %=============station to train===========================
        if  ColumnPool(r,2)==ColumnPool(k,2) && ColumnPool(r,1)==ColumnPool(k,1)
        Flow= FlowStationTrainMatrix(ColumnPool(k,1),ColumnPool(k,2));
        Time=TimeStaTraMatrix(ColumnPool(k,1),ColumnPool(k,2));
        [~,TT]=BPRDev(Flow,Time,Alpha(1),Belta(1),CapacityFacility(1));
        GC=GC+TT*ceil(DecayRouteMatrix(1,k))*DecayRouteMatrix(1,r);
        % incidence(a,p) = ceil(DecayRouteMatrix(4,k)) (inputs are 0 or 1)
        end
        AdditionCost=AdditionCost+GC*lang;
%         if k==r
%             AdditionCost=AdditionCost+PPP(r)*lang;
%         end
    end
    if AdditionCost~=0
    1;
    end
    GeneralizedCost(r)=AdditionCost;
end
% if sum(GeneralizedCost)>0
% 1;
% end


end