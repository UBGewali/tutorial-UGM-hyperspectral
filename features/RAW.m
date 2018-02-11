classdef RAW < handle
	properties
		maps;	
	end
	methods
		function obj = RAW(image)
			obj.maps{1}=image;
		end
		function n = length(obj)
			n = length(obj.maps);
		end
		function amap = get(obj,n)
			amap = obj.maps{n};
		end
	end
end



