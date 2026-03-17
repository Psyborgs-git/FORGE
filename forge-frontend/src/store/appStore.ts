import { create } from 'zustand';
import type { HardwareProfile } from '@/types';

interface AppState {
  mode: 'guided' | 'expert';
  activeProjectId: string | null;
  activeModule: string;
  hardwareProfile: HardwareProfile | null;

  setMode: (mode: 'guided' | 'expert') => void;
  setActiveProjectId: (id: string | null) => void;
  setActiveModule: (module: string) => void;
  setHardwareProfile: (profile: HardwareProfile | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  mode: 'guided',
  activeProjectId: null,
  activeModule: 'playground',
  hardwareProfile: null,

  setMode: (mode) => set({ mode }),
  setActiveProjectId: (id) => set({ activeProjectId: id }),
  setActiveModule: (module) => set({ activeModule: module }),
  setHardwareProfile: (profile) => set({ hardwareProfile: profile }),
}));
